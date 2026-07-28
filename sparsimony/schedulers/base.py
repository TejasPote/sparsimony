from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseScheduler(ABC):

    def __init__(
        self,
        quantity: float,
        t_end: int,
        delta_t: int,
    ):
        self.quantity = quantity
        self.t_end = t_end
        self.delta_t = delta_t

    def next_step_update(self, last_step: int) -> bool:
        if (last_step + 1) % self.delta_t == 0:
            return True
        return False

    @abstractmethod
    def __call__(self, step: int) -> Optional[float]: ...


class StaticScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        return None


class AlwaysTrueScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        return True


class ConstantScheduler(BaseScheduler):

    def __init__(
        self, quantity: float, t_end: int, delta_t: int, *args, **kwargs
    ):
        super().__init__(quantity, t_end, delta_t)

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        else:
            return self.quantity


class CosineDecayScheduler(BaseScheduler):

    def __init__(
        self, quantity: float, t_end: int, delta_t: int, *args, **kwargs
    ):
        super().__init__(quantity, t_end, delta_t)

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        else:
            return self.quantity / 2 * (1 + np.cos((step * np.pi) / self.t_end))


class DenseToSparseCosineScheduler(BaseScheduler):
    """CosineDecay schedule with an initial dense-training phase.

    Returns ``None`` (no topology update) for every step before ``t_dense`` so
    the model trains fully dense during the warmup. From ``t_dense`` onward it
    behaves like ``CosineDecayScheduler`` but **re-anchored to**
    ``[t_dense, t_end]``: the prune-ratio decays from ``quantity`` at
    ``t_dense`` to ~0 at ``t_end``. The first non-``None`` value is returned at
    the first ``delta_t`` boundary ``>= t_dense`` (the dense phase therefore ends
    on that boundary).
    """

    def __init__(
        self,
        quantity: float,
        t_end: int,
        delta_t: int,
        t_dense: int,
        *args,
        **kwargs,
    ):
        super().__init__(quantity, t_end, delta_t)
        self.t_dense = t_dense

    def next_step_update(self, last_step: int) -> bool:
        # Suppress dense-grad accumulation (RigL_D2S) during the dense phase.
        if last_step + 1 < self.t_dense:
            return False
        return (last_step + 1) % self.delta_t == 0

    def __call__(self, step: int) -> Optional[float]:
        if step < self.t_dense:
            return None
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        denom = max(self.t_end - self.t_dense, 1)
        return (
            self.quantity
            / 2
            * (1 + np.cos(((step - self.t_dense) * np.pi) / denom))
        )


class OneShotSparsityScheduler(BaseScheduler):
    """Return ``final_sparsity`` from ``t_prune`` onward, ``None`` before it.

    Drives a single dense -> sparse transition for static (prune-once) sparse
    training: the sparsifier trains fully dense while this returns ``None``,
    then prunes once to ``final_sparsity`` and freezes the topology for the rest
    of training.

    Like ``AcceleratedCubicScheduler`` (and unlike the cosine schedulers) the
    returned value is an absolute sparsity level, not a prune ratio.

    Stateless by design: it keeps returning ``final_sparsity`` after ``t_prune``
    and the sparsifier holds the fire-once latch. Re-pruning an already-pruned
    mask to the same level is a no-op (``calculate_n_drop`` returns <= 0), so
    this is safe across checkpoint resume.
    """

    def __init__(self, final_sparsity: float, t_prune: int, *args, **kwargs):
        super().__init__(quantity=final_sparsity, t_end=t_prune, delta_t=1)
        self.final_sparsity = final_sparsity
        self.t_prune = t_prune

    def next_step_update(self, last_step: int) -> bool:
        # No dense gradient accumulation needed (plain FakeSparsity).
        return False

    def __call__(self, step: int) -> Optional[float]:
        if step < self.t_prune:
            return None
        return self.final_sparsity


class SoftMemoryBoundScheduler(BaseScheduler):
    def __init__(
        self,
        quantity: float,
        t_end: int,
        delta_t: int,
        t_grow: int,
        *args,
        **kwargs,
    ):
        super().__init__(quantity, t_end, delta_t)
        self.t_grow = t_grow
        assert t_grow < delta_t

    def next_step_update(self, last_step: int) -> bool:
        if last_step + 1 > self.t_end:
            return False
        if last_step % self.delta_t == (self.delta_t - self.t_grow):
            # start filling buffers for grow step
            return True
        # elif last_step % self.delta_t == 0:
        #     return True
        # elif (
        #     last_step % self.delta_t == self.t_grow and last_step > self.delta_t  # noqa
        # ):
        #     # Prune next step (need plus one?)
        #     return True
        return False

    def __call__(self, step: int) -> Optional[float]:
        if step > self.t_end:
            return None
        if step % self.delta_t == 0:
            return -self.quantity  # Grow by prune ratio
        elif step % self.delta_t == self.t_grow and step > self.delta_t:
            return self.quantity  # Prune by prune ratio
        else:
            return None


class AcceleratedCubicScheduler(BaseScheduler):
    def __init__(
        self,
        t_end: int,
        delta_t: int,
        t_accel: int,
        initial_sparsity: float = 0.0,
        accelerated_sparsity: float = 0.7,
        final_sparsity: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__(None, t_end, delta_t)
        self.t_accel = t_accel
        self.initial_sparsity = initial_sparsity
        self.accelerated_sparsity = accelerated_sparsity
        self.final_sparsity = final_sparsity

    def __call__(self, step: int) -> Optional[float]:
        if step > self.t_end:
            return None
        elif step % self.delta_t != 0:
            return None
        else:  # Prune
            if step < self.t_accel:
                return self.initial_sparsity
            else:
                return (
                    self.final_sparsity
                    + (self.accelerated_sparsity - self.final_sparsity)
                    * (1 - (step - self.t_accel) / self.t_end) ** 3
                )
