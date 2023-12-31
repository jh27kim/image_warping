__version__ = "0.14.0"

from diffusers.configuration_utils import ConfigMixin
from diffusers.utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_inflect_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_librosa_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    logging,
)


try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_onnx_objects import *  # noqa F403
else:
    from diffusers.pipelines import OnnxRuntimeModel

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_pt_objects import *  # noqa F403
else:
    from .models import (
        AutoencoderKL_Pretrained,
        AutoencoderKL_ControlNet,
        ControlNetModel,
        ModelMixin,
        PriorTransformer,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        VQModel,
    )
    from diffusers.optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from diffusers.pipelines import (
        AudioPipelineOutput,
        DanceDiffusionPipeline,
        DDIMPipeline,
        DDPMPipeline,
        DiffusionPipeline,
        DiTPipeline,
        ImagePipelineOutput,
        KarrasVePipeline,
        LDMPipeline,
        LDMSuperResolutionPipeline,
        PNDMPipeline,
        RePaintPipeline,
        ScoreSdeVePipeline,
    )
    from diffusers.schedulers import (
        DDIMInverseScheduler,
        DDIMScheduler,
        DDPMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        UnCLIPScheduler,
        UniPCMultistepScheduler,
        VQDiffusionScheduler,
    )
    from diffusers.training_utils import EMAModel

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_torch_and_scipy_objects import *  # noqa F403
else:
    from diffusers.schedulers import LMSDiscreteScheduler


try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from diffusers.pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        CycleDiffusionPipeline,
        LDMTextToImagePipeline,
        PaintByExamplePipeline,
        SemanticStableDiffusionPipeline,
        StableDiffusionAttendAndExcitePipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionPanoramaPipeline,
        StableDiffusionPipeline,
        StableDiffusionPipelineSafe,
        StableDiffusionPix2PixZeroPipeline,
        StableDiffusionSAGPipeline,
        StableDiffusionUpscalePipeline,
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
        UnCLIPImageVariationPipeline,
        UnCLIPPipeline,
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
        VQDiffusionPipeline,
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    from diffusers.pipelines import StableDiffusionKDiffusionPipeline

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    from diffusers.pipelines import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        StableDiffusionOnnxPipeline,
    )

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    from diffusers.pipelines import AudioDiffusionPipeline, Mel

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_flax_objects import *  # noqa F403
else:
    from .models.modeling_flax_utils import FlaxModelMixin
    from .models.unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .models.vae_flax import FlaxAutoencoderKL
    from diffusers.pipelines import FlaxDiffusionPipeline
    from diffusers.schedulers import (
        FlaxDDIMScheduler,
        FlaxDDPMScheduler,
        FlaxDPMSolverMultistepScheduler,
        FlaxKarrasVeScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxPNDMScheduler,
        FlaxSchedulerMixin,
        FlaxScoreSdeVeScheduler,
    )


try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
    from diffusers.pipelines import (
        FlaxStableDiffusionImg2ImgPipeline,
        FlaxStableDiffusionInpaintPipeline,
        FlaxStableDiffusionPipeline,
    )
