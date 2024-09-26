def train(model, cfg, device):
    wandb.init(project='3ball_CAP', entity='hails', config={
        "batch_size": cfg.batch_size,
        "overshooting_distance": cfg.overshooting_distance,
        "episodes": cfg.episodes,
        "chunk_size": cfg.chunk_size,
        "planning_horizon": cfg.planning_horizon,
        "planning_discount": cfg.discount,
        "total_episodes" : cfg.total_steps,
        "max_steps": cfg.max_steps,
    })
    print("======== Settings ========")
    pprint(cfg)
    print("======== Model ========")
    pprint(model)

    model = model.to(device)

    # Optimizer 및 체크포인터 설정
    optimizers = get_optimizer(cfg, model)
    checkpointer_path = os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
    checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num)

    # 설정 파일 저장
    with open(checkpointer_path + '/config.yaml', 'w') as f:
      cfg.dump(stream=f, default_flow_style=False)
      print(f"config file saved to {checkpointer_path + '/config.yaml'}")

    # 체크포인트 로드 (있을 경우)
    if cfg.resume:
      checkpoint = checkpointer.load(cfg.resume_ckpt)
      if checkpoint:
        model.load_state_dict(checkpoint['model'])
        for k, v in optimizers.items():
          if v is not None:
            v.load_state_dict(checkpoint[k])
        env_step = checkpoint['env_step']
        global_step = checkpoint['global_step']
      else:
        env_step = 0
        global_step = 0
    else:
      env_step = 0
      global_step = 0

    # TensorBoard 설정
    writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id), flush_secs=30)

    # 환경 설정 및 초기화
    datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'train_episodes')
    test_datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'test_episodes')
    train_env = make_env(cfg, writer, 'train', datadir, store=True)
    test_env = make_env(cfg, writer, 'test', test_datadir, store=True)

    # 초기 프리필 단계에서 랜덤 행동으로 데이터 수집
    train_env.reset()
    steps = count_steps(datadir, cfg)
    length = 0
    while steps < cfg.arch.prefill:
      action = train_env.sample_random_action()
      next_obs, reward, done = train_env.step(action[0])
      length += 1
      steps += done * length
      length = length * (1. - done)
      if done:
        train_env.reset()

    steps = count_steps(datadir, cfg)
    print(f'collected {steps} steps. Start training...')
    train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
    train_iter = iter(train_dl)
    global_step = max(global_step, steps)

    obs = train_env.reset()
    state = None
    action_list = torch.zeros(1, 1, cfg.env.action_size).float()  # T, C
    action_list[0, 0, 0] = 1.0
    input_type = cfg.arch.world_model.input_type
    temp = cfg.arch.world_model.temp_start
    episode_num = 0  # 에피소드 카운터

    train_steps = 0
    total_train_steps = 0
    # 에피소드 단위 학습 루프
    while episode_num < cfg.total_steps:
      with torch.no_grad():
        model.eval()
        total_reward = 0  # 현재 에피소드의 총 보상
        done = False
        print("episode is now :", episode_num)
        while not done:
          train_steps = 0
          # 환경과 상호작용
          next_obs, reward, done = train_env.step(action_list[0, -1].detach().cpu().numpy())
          train_steps += 1
          prev_image = torch.tensor(obs[input_type])
          next_image = torch.tensor(next_obs[input_type])

          # 모델의 정책에 따라 행동 선택
          action_list, state = model.policy(
            prev_image.to(device), next_image.to(device), action_list.to(device),
            global_step, 0.1, state, context_len=cfg.train.batch_length
          )
          total_train_steps += train_steps
          total_reward += reward
          obs = next_obs
          if done:
            print("total_train_steps per episode : ", train_steps)
            train_env.reset()
            state = None
            action_list = torch.zeros(1, 1, cfg.env.action_size).float()
            action_list[0, 0, 0] = 1.0

      # 에피소드 종료 후 학습 진행
      model.train()
      traj = next(train_iter)
      for k, v in traj.items():
        traj[k] = v.to(device).float()

      logs = {}

      # 모델 및 옵티마이저 업데이트
      model_optimizer = optimizers['model_optimizer']
      model_optimizer.zero_grad()
      transformer_optimizer = optimizers['transformer_optimizer']
      if transformer_optimizer is not None:
        transformer_optimizer.zero_grad()

      model_loss, model_logs, prior_state, post_state = model.world_model_loss(global_step, traj, temp)
      grad_norm_model = model.world_model.optimize_world_model(model_loss, model_optimizer, transformer_optimizer,
                                                               writer, global_step)

      if cfg.arch.world_model.transformer.warm_up:
        lr = anneal_learning_rate(global_step, cfg)
        for param_group in transformer_optimizer.param_groups:
          param_group['lr'] = lr
      else:
        lr = cfg.optimize.model_lr

      actor_optimizer = optimizers['actor_optimizer']
      value_optimizer = optimizers['value_optimizer']
      actor_optimizer.zero_grad()
      value_optimizer.zero_grad()

      # Actor와 Value의 손실 계산 및 최적화
      actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(global_step, post_state, traj, temp)
      grad_norm_actor = model.optimize_actor(actor_loss, actor_optimizer, writer, global_step)
      grad_norm_value = model.optimize_value(value_loss, value_optimizer, writer, global_step)

      wandb.log({
          "episode": episode_num,
          "steps": train_steps,
          "reward": total_reward,
          "model_loss": model_loss.item(),  # 모델 손실 기록
          "actor_loss": actor_loss.item(),  # Actor 손실 기록
          "value_loss": value_loss.item(),  # Value 손실 기록
          # "mean_value": np.mean(episode_values),  # 필요시 평균 값 기록
      }, step=episode_num)

      # 로그 기록
      if global_step % cfg.train.log_every_step == 0:
        logs.update(model_logs)
        logs.update(actor_value_logs)
        model.write_logs(logs, traj, global_step, writer)
        writer.add_scalar('train_hp/lr', lr, global_step)

        grad_norm = dict(
          grad_norm_model=grad_norm_model,
          grad_norm_actor=grad_norm_actor,
          grad_norm_value=grad_norm_value,
        )

        for k, v in grad_norm.items():
          writer.add_scalar('train_grad_norm/' + k, v, global_step=global_step)

      # 주기적인 평가 및 체크포인트 저장
      if global_step % cfg.train.eval_every_step == 0:
        simulate_test(model, test_env, cfg, global_step, device)

      if global_step % cfg.train.checkpoint_every_step == 0:
        env_step = count_steps(datadir, cfg)
        checkpointer.save('', model, optimizers, global_step, env_step)

      # 에피소드 수 증가
      episode_num += 1
      global_step += 1

    writer.close()
    wandb.close()