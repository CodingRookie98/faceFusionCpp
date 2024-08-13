## Installation

If you are using GPU acceleration, please install the CUDA environment. If TensorRT acceleration is enabled, please install TensorRT.

 Version requirements:

>   **CUDA >= 12.2 (tested with CUDA 12.5)**

>   **CUDNN >= 9.2 (tested)**

>   **TensorRT >= 10.2 (tested) (if enabled)**
>
>   **ffmpeg >= 7.0.2 (tested)(for video)**



## Configuration File

### general

#### source_path

Source path, **currently supports images only**. Accepts a string that can be either a single file path or a directory path containing multiple files.

>   [!NOTE]
>
>   The source path is not required when performing facial enhancement only.

#### target_path

 Target path, **currently supports images only**. Accepts a string that can be either a single file path or a directory path containing multiple files.

#### reference_face_path

Path to the reference face, **currently supports images only**. Accepts a string; currently supports only a single image, with future plans to support multiple images as input.

#### output_path

Path for storing output files, supports both file paths and directory paths, with directory paths recommended.

>   [!WARNING]
>
>    When there are multiple targets but a file path is used as the output path, output files may be overwritten, resulting in only one output for multiple targets.



### misc

#### force_download

Force automate downloads and exit.

```
default: true
choice: false true
```

#### skip_download

Omit automate downloads and remote lookups.

```
default: false
choices: false true
```

#### log_level

Adjust the message severity displayed in the terminal.

```
Default: info
choice: trace debug info warn error critical
```



### execution

#### execution_device_id

Specify the device used for processing.

```
default: 0
```

#### execution_providers
>
Accelerate model inference using different providers.
>
```
default: cpu
choices: tensorrt cuda cpu
```
>
Multiple providers can be selected, with priority as follows: tensorrt > cuda > cpu.
>
#### execution_thread_count
>
This parameter specifies the number of threads for inference tasks. If GPU memory is limited, consider reducing this value.
>
```
default: 1
```


### tensorrt

#### enable_engine_cache

Enable TensorRT engine caching. This option is effective only when using the TensorRT engine.

```
default: false
choices: false true
```
>   [!TIP]
>
>   Enabling this option will use some disk space but will avoid the model warm-up time each time TensorRT is activated, thus speeding up the startup. It is recommended to enable this when using TensorRT.

#### enable_embed_engine


Get the embedded engine model via a warmup run with the original model.

```
default: false
choices: false true
```
>   [!TIP]
>
>   It is recommended to enable this option when using TensorRT.



### memory

#### per_session_gpu_mem_limit

The amount of GPU memory allocated per session, in GB. Both decimal and integer values are acceptable.

```
example: 2
```
>   [!NOTE]
>
>   This option is useful for devices with low GPU memory. For high-end GPUs, it may not be as relevant, as my device is a six-year-old laptop (GTX1650, 4GB GPU RAM).



### face_analyser

#### face_detector_model

```
Default: yoloface
Choices: many retinaface scrfd yoloface yunet
```
>   [!NOTE]
>
>   The "many" option uses retinaface, scrfd, and yoloface, excluding yunet. Yunet can be enabled separately.

#### face_detector_size

Specify the size of the frame provided to the face detector.

```
Default: 640x640
Choices: 160x160 320x320 480x480 512x512 640x640 768x768 960x960 1024x1024
```

#### face_detector_score

Filter the detected faces base on the confidence score.

```
Default: 0.5
Range: 0 to 1 at 0.05
Example: 0.7
```

#### face_landmarker_score

Filter the detected landmarks base on the confidence score.

```
Argument: --face-landmarker-score
Default: 0.5
Range: 0 to 1 at 0.05
Example: 0.7
```



### face_selector

#### face_selector_mode

Use reference based tracking or simple matching.

```
Default: many
Choices: many one reference
Example: --face-selector-mode one
```

####  face_selector_order

Specify the order in which the face analyser detects faces.

```
Default: left-right
Choices: left-right right-left top-bottom bottom-top small-large large-small best-worst worst-best
Example: best-worst
```

#### face_selector_age

Filter the detected faces based on their age.

```
Default: None
Choices: child teen adult senior
Example: --face-selector-age adult
```

#### face_selector_gender

Filter the detected faces based on their gender.

```
Default: None
Choices: male female
Example: male
```

#### reference_face_position

Specify the position used to create the reference face.

```
Default: 0
Example: 1
```

#### reference_face_distance

Specify the desired similarity between the reference face and target face.

```
Default: 0.6
Range: 0 to 1.5 at 0.05
Example: 0.8
```



### face_mask

#### face_mask_types

Mix and match different face mask types.

```
Default: box
Choices: box occlusion region
Example: box occlusion
```

#### face_mask_blur

Specify the degree of blur applied the box mask.

```
Default: 0.3
Range: 0 to 1 at 0.05
Example: 0.6
```

#### face_mask_padding

Apply top, right, bottom and left padding to the box mask.

```
Default: 0 0 0 0
Example: 1 2
```

#### face_mask_regions

Choose the facial features used for the region mask.

```
Default: All
Choices: skin left-eyebrow right-eyebrow left-eye right-eye glasses nose mouth upper-lip lower-lip
Example: left-eye right-eye eye-glasses
```



### image

#### output_image_quality

Specify the image quality which translates to the compression factor.

```
Default: 100
Range: 0 to 100 at 1
Example: 60
```

#### output_image_resolution

Specify the image output resolution based on the target image.

```
Default: None
Example: 1920x1080
```



### video

#### video_segment_duration

If this configuration item is greater than 0, the target video will be cut into multiple segments for processing. The duration of each segment will be the value of this item, in seconds.

```
default: 0
example: 30
```

>   [!TIP]
>
>   When processing long videos, it's recommended to handle them in segments. This approach can reduce memory and disk usage, although it may slightly decrease processing speed.

#### output_video_encoder

Specify the encoder used for the video output.

```
Default: libx264
Choices: libx264 libx265 libvpx-vp9 h264_nvenc hevc_nvenc h264_amf hevc_amf
Example: libx265
```

#### output_video_preset

Balance fast video processing and video file size.

```
Default: veryfast
Choices: ultrafast superfast veryfast faster fast medium slow slower veryslow
Example: faster
```

#### output_video_quality

Specify the video quality which translates to the compression factor.

```
Default: 80
Range: 0 to 100 at 1
Example: 60
```

#### output_audio_encoder

Specify the encoder used for the audio output.

```
Default: aac
Choices: aac libmp3lame libopus libvorbis
Example: libmp3lame
```

#### skip_audio

Omit the audio from the target video.

```
Default: false
choice: true false
Example: true
```

#### temp_frame_format

Specify the format in which the extracted video frames should be stored as images.

```
default: png
choice: png jpg bmp
```



### frame_processors

#### frame_processors

Load a single or multiple frame processors.

```
Default: face_swapper
Choices: face_enhancer face_swapper
Example: face_swapper face_enhancer
```

#### face_enhancer_model

Choose the model responsible for enhancing the face.

```
Default: gfpgan_1.4
Choices: codeformer gfpgan_1.2 gfpgan_1.3 gfpgan_1.4 gpen_bfr_256 gpen_bfr_512 gpen_bfr_1024 gpen_bfr_2048 restoreformer_plus_plus
Example: codeformer
```

#### face_enhancer_blend

Blend the enhanced into the previous face.

```
Default: 80
Range: 0 to 100 at 1
Example: 60
```

#### face_swapper_model

Choose the model responsible for swapping the face.

```
Default: inswapper_128_fp16
Choices: blendswap_256 ghost_256_unet_1 ghost_256_unet_2 ghost_256_unet_3 inswapper_128 inswapper_128_fp16 simswap_256 simswap_512_unofficial uniface_256
Example: simswap_256
```