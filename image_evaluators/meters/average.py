from torch import Tensor


class AverageMeter:
    def __init__(self):
        self.sum = None
        self.count = 0

    def update(self, values: Tensor):
        assert values.ndim == 1
        values_sum = values.sum().float()
        values_count = values.shape[0]
        self.sum = values_sum if self.sum is None else self.sum + values_sum
        self.count = self.count + values_count

    @property
    def average(self) -> Tensor:
        return self.sum / self.count
