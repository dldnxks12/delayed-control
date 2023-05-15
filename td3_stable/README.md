  
      modify sampling time :
      
        ./environment/inverted_single_pendulum_v4.py -> frame_skip, render_fps, self.dt check
        ./environment/assets/inverted_single_pendulum.xml -> time check 
        
               

      On custom inverted pendulum :
      
        sampling time 20ms -> 12 sample --- 2.4s
        sampling time 50ms -> 5 sample  --- 2.5s
        
        # hyperparameters #
        
        self.batch_size      = 128
        self.act_noise_scale = 0.1
        self.actor_lr        = 0.001
        self.critic_lr       = 0.001
        self.gamma           = 0.99
        self.tau             = 0.005
