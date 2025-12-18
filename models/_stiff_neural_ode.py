'''
Author: Hongjue Zhao (Modified for Stiff ODE Experiments)
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

from typing import Callable, Union, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
from jaxtyping import Float, Array, Key

from ._nde_solver import SolverKwargs, generate_nde_solver_fn


class StiffODE_Func(eqx.Module):
    '''
    Vector field function for Stiff Neural ODEs with Output Scaling.
    
    Reference: 
        - Kim et al., "Stiff Neural Ordinary Differential Equations", 2021
    
    Args:
        - `key`: `Key` - Random key
        - `obs_size`: `int` - Observation size
        - `width_size`: `int` - Width size
        - `depth`: `int` - Depth
        - `activation`: `Callable` - Activation function
        - `final_activation`: `Callable` - Final activation function
        - `scale`: `Array` - Scaling factors for state variables (optional)
    '''
    mlp: eqx.nn.MLP
    scale: Array
    
    def __init__(
        self, 
        key: Key, 
        obs_size: int, 
        width_size: int = 64, 
        depth: int = 4, 
        activation: Callable = jax.nn.elu, 
        final_activation: Callable = lambda x: x,
        scale: Optional[Array] = None
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            obs_size, obs_size, width_size, depth, 
            activation, final_activation, key = key
        )
        
        # 如果没有提供 scale，默认为全 1 (退化为普通 NeuralODE)
        if scale is None:
            self.scale = jnp.ones(obs_size)
        else:
            self.scale = jnp.asarray(scale)
        
    def __call__(self, t, y, args = None):
        # 1. Input Scaling: 将物理状态 y 归一化 (y_hat = y / s)
        # 为了数值稳定性，最好在传入 scale 前确保其不为 0
        y_norm = y / self.scale
        
        # 2. Network Prediction: 网络预测归一化后的导数 dy_hat/dt
        dy_norm = self.mlp(y_norm)
        
        # 3. Output Scaling: 将预测还原为物理导数
        # dy/dt = d(y_norm * s)/dt = dy_norm/dt * s
        return dy_norm * self.scale


class StiffNeuralODE(eqx.Module):
    '''
    Stiff Neural ODE model wrapper.
    Designed to handle stiff dynamics like Robertson or POLLU problems.
    '''
    ode_func: StiffODE_Func
    solver_fn: Callable
    
    def __init__(
        self, 
        key: Key, 
        obs_size: int, 
        width_size: int = 64, 
        depth: int = 4, 
        activation: Callable = jax.nn.elu, 
        final_activation: Callable = lambda x: x,
        solver_kws: SolverKwargs = SolverKwargs(),
        scale: Optional[Array] = None  # 新增接口用于接收缩放因子
    ):
        super().__init__()
        # 初始化带有 Scaling 功能的 ODE Func
        self.ode_func = StiffODE_Func(
            key, obs_size, width_size, depth, 
            activation, final_activation, 
            scale=scale
        )
        # 使用刚性求解器配置生成求解函数
        self.solver_fn = generate_nde_solver_fn(solver_kws)
            
    @eqx.filter_jit
    def __call__(
        self, 
        ts: Float[Array, 'tspan'], 
        y0: Float[Array, 'obs'],
        key: Key = None,
    ):
        sol = self.solver_fn(
            terms = dfx.ODETerm(self.ode_func), y0 = y0,
            t0 = ts[0], t1 = ts[-1], saveat = dfx.SaveAt(ts = ts)
        )
        return sol.ys
    
    def vector_field(self, ts, ys):
        return jax.vmap(self.ode_func)(ts, ys)