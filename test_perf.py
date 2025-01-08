import torch
from line_profiler import LineProfiler
import atexit

profile = LineProfiler()
atexit.register(profile.print_stats)

@profile
def test_perf():
    device = torch.device('cuda:1')
    for _ in range(1000):
        aa = 2 * torch.ones((10, 1024, 1024), dtype=torch.float32)
        bb = aa * aa
        cc = aa.to(device)
        dd = cc * cc
        ee = dd.cpu()

if __name__ == '__main__':
    print(f'torch version: {torch.__version__}')
    print(f'cuda version: {torch.version.cuda}')
    test_perf()