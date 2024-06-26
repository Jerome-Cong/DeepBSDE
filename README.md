<!--
 * @Description: 
 * @Version: 1.0
 * @Autor: Shijie Cong
 * @Date: 2024-01-05 14:16:31
 * @LastEditors: Shijie Cong
 * @LastEditTime: 2024-01-16 17:38:28
-->
# [Deep BSDE Solver](https://doi.org/10.1073/pnas.1718942115) in TensorFlow/Pytorch

## Usage

- I recommend doing installation in a virtual environment such as [Anaconda](https://www.anaconda.com/), Python 3.9 is used in this implementation.
- Install [PyTorch](https://pytorch.org/get-started/locally/)

```bash
conda install cudatoolkit=11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Clone this repository

```bash
git clone https://github.com/Jerome-Cong/DeepBSDE.git
pip3 install -e .
```


## Training

- With TensorFlow:

```bash
python main.py --config_path=configs/hjb_lq_d100.json
```

- With PyTorch:

```bash
python main_torch.py --config_path=configs/hjb_lq_d100.json
```

Command-line flags:

* `config_path`: Config path corresponding to the partial differential equation (PDE) to solve. 
There are seven PDEs implemented so far. See [Problems](#problems) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.
* `log_dir`: Directory to write logging and output array.


## Problems

`equation.py` and `config.py` now support the following problems:

Three examples in ref [1]:
* `HJBLQ`: Hamilton-Jacobi-Bellman (HJB) equation.
* `AllenCahn`: Allen-Cahn equation with a cubic nonlinearity.
* `PricingDefaultRisk`: Nonlinear Black-Scholes equation with default risk in consideration.


Four examples in ref [2]:
* `PricingDiffRate`: Nonlinear Black-Scholes equation for the pricing of European financial derivatives
with different interest rates for borrowing and lending.
* `BurgersType`: Multidimensional Burgers-type PDEs with explicit solution.
* `QuadraticGradient`: An example PDE with quadratically growing derivatives and an explicit solution.
* `ReactionDiffusion`: Time-dependent reaction-diffusion-type example PDE with oscillating explicit solutions.


New problems can be added very easily. Inherit the class `equation`
in `equation.py` and define the new problem. Note that the generator function 
and terminal function should be TensorFlow operations while the sample function
can be python operation. A proper config is needed as well.

## Reference

[1] Han, J., Jentzen, A., and E, W. Overcoming the curse of dimensionality: Solving high-dimensional partial differential equations using deep learning,
<em>Proceedings of the National Academy of Sciences</em>, 115(34), 8505-8510 (2018). [[journal]](https://doi.org/10.1073/pnas.1718942115) [[arXiv]](https://arxiv.org/abs/1707.02568) <br />
[2] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations,
<em>Communications in Mathematics and Statistics</em>, 5, 349–380 (2017). 
[[journal]](https://doi.org/10.1007/s40304-017-0117-6) [[arXiv]](https://arxiv.org/abs/1706.04702)


