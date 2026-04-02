"""
SegviGen flow sampler wrapper.

Thin wrapper around FlowEulerGuidanceIntervalSampler that properly forwards
guidance_rescale and rescale_t (both silently dropped in current code).
"""
import logging

log = logging.getLogger("segvigen")


class SegviGenFlowSampler:
    """
    Wrapper that forwards guidance_rescale and rescale_t to the underlying
    TRELLIS2 FlowEulerGuidanceIntervalSampler.
    """

    def __init__(self, sigma_min: float = 1e-5):
        self._sigma_min = sigma_min

    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int,
        guidance_strength: float,
        guidance_rescale: float = 0.0,
        guidance_interval: tuple = (0.6, 0.9),
        rescale_t: float = 1.0,
        **extra_model_kwargs,
    ):
        from trellis2.samplers import FlowEulerGuidanceIntervalSampler

        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=self._sigma_min)
        return sampler.sample(
            model, noise,
            cond=cond,
            neg_cond=neg_cond,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
            verbose=True,
            **extra_model_kwargs,
        )
