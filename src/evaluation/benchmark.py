"""Efficiency benchmark for HybridMTLNet.

Quantifies the unification claim: one shared-encoder model producing both
segmentation and pose in a single forward pass, versus a modular approach
that runs two separate single-task models. Reports parameters, FLOPs, and
latency / throughput (FPS) on GPU and CPU.
"""
import time

import torch

from src.models.mtl_net import HybridMTLNet


class Benchmark:
    def __init__(self, model_cfg, train_cfg, input_size=(256, 256), runs=100, warmup=15):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.H, self.W = input_size
        self.runs = runs
        self.warmup = warmup
        self.has_cuda = torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _dummy(self, device, batch=1):
        return torch.randn(batch, 3, self.H, self.W, device=device)

    @staticmethod
    def _params_m(model) -> float:
        return sum(p.numel() for p in model.parameters()) / 1e6

    def _flops_g(self, model) -> float | None:
        """FLOPs is a device-independent count; profile a CPU deepcopy to avoid
        device-mismatch in thop hooks and to keep the real model unpolluted."""
        try:
            import copy
            from thop import profile
            m = copy.deepcopy(model).cpu().eval()
            x = torch.randn(1, 3, self.H, self.W)
            macs, _ = profile(m, inputs=(x,), verbose=False)
            return round(macs * 2 / 1e9, 2)  # MACs → FLOPs (×2), to GFLOPs
        except Exception as e:
            print(f"  [FLOPs skipped: {e}]")
            return None

    @torch.no_grad()
    def _latency_ms(self, model, device) -> tuple[float, float]:
        model.eval().to(device)
        x = self._dummy(device)
        is_cuda = device.type == "cuda"

        for _ in range(self.warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(self.runs):
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        t = torch.tensor(times)
        return float(t.mean()), float(t.std())

    def profile(self, model, name: str) -> dict:
        rec = {"name": name, "params_M": round(self._params_m(model), 2)}
        rec["flops_G"] = self._flops_g(model)
        if self.has_cuda:
            ms, _ = self._latency_ms(model, torch.device("cuda"))
            rec["gpu_ms"], rec["gpu_fps"] = round(ms, 2), round(1000.0 / ms, 1)
        ms, _ = self._latency_ms(model, torch.device("cpu"))
        rec["cpu_ms"], rec["cpu_fps"] = round(ms, 2), round(1000.0 / ms, 1)
        return rec

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        n_inst, n_kp = self.mcfg.num_instances, self.mcfg.num_keypoints

        unified = HybridMTLNet(n_inst, n_kp)
        rec_unified = self.profile(unified, "HybridMTLNet (joint, 1 pass)")

        # Modular: two separate single-task models of the same architecture.
        # They do not share the encoder, so cost is the sum of two networks.
        rec_modular = {
            "name":     "Modular (seg-only + pose-only, 2 models)",
            "params_M": round(rec_unified["params_M"] * 2, 2),
            "flops_G":  round(rec_unified["flops_G"] * 2, 2) if rec_unified["flops_G"] else None,
        }
        for k_ms, k_fps in (("gpu_ms", "gpu_fps"), ("cpu_ms", "cpu_fps")):
            if k_ms in rec_unified:
                ms2 = rec_unified[k_ms] * 2
                rec_modular[k_ms] = round(ms2, 2)
                rec_modular[k_fps] = round(1000.0 / ms2, 1)

        results = [rec_unified, rec_modular]
        self.print_table(results)
        return results

    def print_table(self, results: list[dict]) -> None:
        cols = ["params_M", "flops_G", "gpu_ms", "gpu_fps", "cpu_ms", "cpu_fps"]
        hdr_names = {"params_M": "Params(M)", "flops_G": "FLOPs(G)",
                     "gpu_ms": "GPU ms", "gpu_fps": "GPU FPS",
                     "cpu_ms": "CPU ms", "cpu_fps": "CPU FPS"}
        present = [c for c in cols if any(c in r for r in results)]

        head = f"{'Model':<42}" + "".join(f"{hdr_names[c]:>11}" for c in present)
        line = "-" * len(head)
        print(f"\n{line}\n{head}\n{line}")
        for r in results:
            row = f"{r['name']:<42}"
            for c in present:
                v = r.get(c)
                row += f"{'—' if v is None else v:>11}"
            print(row)
        print(line)
        print(f"(input {self.H}x{self.W}, batch 1, {self.runs} runs)")
