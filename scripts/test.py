import logging
import os
import sys

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

# 设置环境和路径，确保能导入项目中的模块
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# 启用 x64 精度，保持与 main.py 一致
jax.config.update("jax_enable_x64", True)

# 导入所有 ODE 类，以便 eval() 可以找到它们
from data.ode import *
from utils import generate_mask

log = logging.getLogger(__name__)


def plot_debug_trajectories(ts, ys, sampled_ys=None, save_dir=None, title_prefix=""):
    """
    专门用于 Dataset Debug 的绘图函数。
    只绘制 Ground Truth 和（可选的）带噪声观测点，不需要模型预测结果。
    """
    traj_idx = 0  # 默认只画第一条轨迹进行检查
    t = ts[traj_idx]
    y = ys[traj_idx]

    obs_size = y.shape[-1]

    # --- 1. 绘制时序图 (Time Series) ---
    fig_cols = min(obs_size, 3)
    fig_rows = (obs_size + fig_cols - 1) // fig_cols

    fig, axes = plt.subplots(
        fig_rows, fig_cols, figsize=(6 * fig_cols, 4 * fig_rows), squeeze=False
    )
    fig.suptitle(f"{title_prefix} Time Series (Trajectory 0)", fontsize=16)

    flat_axes = axes.flatten()

    for i in range(obs_size):
        ax = flat_axes[i]
        ax.plot(t, y[:, i], label="Ground Truth", color="black", linewidth=2)

        if sampled_ys is not None:
            # 过滤掉 NaN 用于绘图
            sy = sampled_ys[traj_idx, :, i]
            st = t
            mask = ~jnp.isnan(sy)
            ax.scatter(
                st[mask],
                sy[mask],
                label="Noisy Observations",
                color="cornflowerblue",
                s=20,
                alpha=0.7,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel(f"Dim {i}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    # 隐藏多余的子图
    for i in range(obs_size, len(flat_axes)):
        flat_axes[i].axis("off")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_timeseries.png"))
    plt.close()

    # --- 2. 绘制相图 (Phase Portrait) - 仅当维度为 2 或 3 时 ---
    if obs_size == 2:
        fig = plt.figure(figsize=(8, 6))
        plt.title(f"{title_prefix} Phase Portrait")
        plt.plot(y[:, 0], y[:, 1], color="black", label="Ground Truth")
        if sampled_ys is not None:
            sy = sampled_ys[traj_idx]
            mask = ~jnp.isnan(sy).any(axis=1)
            plt.scatter(
                sy[mask, 0],
                sy[mask, 1],
                color="cornflowerblue",
                label="Observations",
                s=20,
            )
        plt.xlabel("Dim 0")
        plt.ylabel("Dim 1")
        plt.legend()
        plt.grid(True, linestyle="--")
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{title_prefix}_phase.png"))
        plt.close()

    elif obs_size == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"{title_prefix} Phase Portrait")
        ax.plot(y[:, 0], y[:, 1], y[:, 2], color="black", label="Ground Truth")
        if sampled_ys is not None:
            sy = sampled_ys[traj_idx]
            mask = ~jnp.isnan(sy).any(axis=1)
            ax.scatter(
                sy[mask, 0],
                sy[mask, 1],
                sy[mask, 2],
                color="cornflowerblue",
                label="Observations",
                s=20,
            )
        ax.set_xlabel("Dim 0")
        ax.set_ylabel("Dim 1")
        ax.set_zlabel("Dim 2")
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{title_prefix}_phase.png"))
        plt.close()


@hydra.main(config_path="../confs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    print(f"=== Debugging Dataset: {cfg.ode.type} ===")

    # 1. 准备输出目录
    output_dir = os.path.join(os.getcwd(), "debug_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")

    # 2. 实例化 ODE 类
    try:
        # 假设你在 data/ode.py 中定义了类，并且已经在此文件头部 import * 了
        ode_class = eval(cfg.ode.type)
        ode_instance = ode_class()
        print(f"Successfully instantiated {cfg.ode.type}")
        print(f"ODE Params: {ode_instance.args}")
        print(f"Initial State Range: {ode_instance.y0_range}")
    except NameError:
        print(f"Error: Class '{cfg.ode.type}' not found via eval().")
        print(
            "Please ensure it is defined in 'data/ode.py' and included in 'data/ode.py' imports."
        )
        return
    except Exception as e:
        print(f"Error instantiating ODE: {e}")
        return

    # 3. 模拟 Ground Truth 数据
    key = jax.random.PRNGKey(cfg.seed)
    sim_key, noise_key, mask_key = jr.split(key, 3)

    print(f"Simulating {cfg.data.traj_num} trajectories with T={cfg.data.T}...")
    ts, ys = ode_instance.simulate(
        sim_key, T=cfg.data.T, point_num=cfg.data.point_num, traj_num=cfg.data.traj_num
    )

    print(f"Data shape: Time {ts.shape}, State {ys.shape}")

    # 4. 模拟观测数据 (添加噪声和 Mask)
    # 模拟 data.generate_single_set 中的逻辑
    noise_std = cfg.data.noise_level * jnp.std(ys, axis=1, keepdims=True)
    noisy_ys = ys + noise_std * jr.normal(noise_key, ys.shape)

    # 生成 mask
    # 注意：这里模拟 data/__init__.py 中的 generate_single_set 逻辑
    # 你的项目中 mask 是通过 split 和 concatenate 完成的，这里简化为直接 mask
    # 这样可以直观看到如果在当前噪声水平和采样率下的数据样子

    # 简单的随机 mask 模拟 (参考 utils.generate_mask)
    if cfg.data.ratio < 1.0:
        mask = generate_mask(mask_key, ys.shape, 1 - cfg.data.ratio)
        sampled_ys = noisy_ys * mask
        sampled_ys = jnp.where(mask == 0, jnp.nan, sampled_ys)
    else:
        sampled_ys = noisy_ys

    # 5. 可视化
    print("Plotting...")
    plot_debug_trajectories(
        ts, ys, sampled_ys, save_dir=output_dir, title_prefix=cfg.ode.type
    )

    print(f"Done! Check the plots in: {output_dir}")


if __name__ == "__main__":
    main()
