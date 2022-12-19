import tensorflow as tf
from text import VOCAB_DICT


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500, # 전채 반복 횟수
        iters_per_checkpoint=1000, # 몇번째 iter마다 저장할지
        seed=1234, # seed cuda, torch 등
        dynamic_loss_scaling=True, # ?
        fp16_run=False, # gpu multi use
        distributed_run=False, # 
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'], # 무시할 layer = lstm embedding vector

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False, # 불러올 melspectogram
        training_files='data/train_filelist.txt',   # 훈련 wave 파일의 경로 | scripts
        validation_files='data/valid_filelist.txt', # 검증 wave 파일의 경로 | scripts

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050, # 몇개의 샘플로 자를지 선택
        filter_length=1024,  # ?
        hop_length=256,      # window에 몇개씩 shift 할지 
        win_length=1024,     # window size
        n_mel_channels=80,   # melspecto gram의 차원 ?
        mel_fmin=0.0,        # min scale ? 
        mel_fmax=8000.0,     # max scale ?

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(VOCAB_DICT.keys()),
        symbols_embedding_dim=512, # ?

        # Encoder parameters
        encoder_kernel_size=5, # conv kernel size
        encoder_n_convolutions=3, # conv kernel 갯수
        encoder_embedding_dim=512, # lstm embedding size ? 

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported 출력되는 melspectogram의 개수 tarcotron2 에서는 성능에 영향이 있기 때문에 1개로 고정함?
        decoder_rnn_dim=1024, # lstm 차원 크기?
        prenet_dim=256, # prenet 크기
        max_decoder_steps=1000, # stop token 1 ?
        gate_threshold=0.5, # stop token 2 ? 
        p_attention_dropout=0.1, # attention drop out
        p_decoder_dropout=0.1, # 

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512, # post net embedding dim
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False, # lr 을 저장할지
        learning_rate=1e-3, # lr 0.0001
        weight_decay=1e-6, # 
        grad_clip_thresh=1.0,
        batch_size=32, # batch size 128(클수록)이 성능이 좋음
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams # hparams return
