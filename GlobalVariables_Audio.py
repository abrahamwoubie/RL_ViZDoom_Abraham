class GlobalVariables_Audio:

    #parameters
    how_many_times_training=1
    how_many_times = 20000
    replay_memory_size = 100000
    replay_memory_batch_size = 64

    Learning_Rate = 0.00025
    Discount_Factor = 0.99

    frame_repeat = 10
    channels = 3

    channels_audio = 1

    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * how_many_times

    save_each = 4000#0.00625 * how_many_times

    final_test_percentage=[]