Inverse Ensemble Forecasting for COVID-19
================

In this repository, we use the [BET](https://ut-chg.github.io/BET/) Python package to model and forecast COVID-19 surges under mutating virus strains. We choose the SIR framework to model disease dynamics and map infection parameters to a Quantity of Interest (QoI) defined by the number of susceptible or infectious individuals in a population at a given time. We assume a probability distribution <!-- $P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\RVfnRrZOeE.svg"> on the infection parameters which propagates through the SIR model and induces a probability distribution <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\70SdivIgEg.svg"> on the QoI. By obtaining population data on infection levels at early stages in a surge, we can observe <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\70SdivIgEg.svg"> and solve the stochastic inverse problem to compute a <!-- $P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\RVfnRrZOeE.svg"> which is consistent with the observed data. This solution can then be used to forecast future infection levels.


## **Theory Overview**

Consider a parameter probability space <!-- $(\Lambda, \mathcal F, P_\Lambda)$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\4nAj1WASk2.svg"> and a function $Q$ which maps <!-- $\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\IrTkO6XxWA.svg"> to an observed output space $D$. The stochastic inverse
problem (SIP) refers to the goal of inferring the probability
distribution <!-- $P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\YVaj3zVrGN.svg"> by observing the induced probability
distribution on <!-- $D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\LLofork7IM.svg">, <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\GiZSnbr7xV.svg">. Given an observed output distribution <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\lc4fUgrELs.svg">,
we seek solutions <!-- $P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\xhWij0QRJS.svg"> satisfying 

$$P_\Lambda(A) = P_D(Q(A))$$

Define <!-- $Q^{-1}\left(B\right) = \{\lambda \in \Lambda : Q(\lambda) \in B\}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\Md2DSufrWi.svg"> for <!-- $B \subset D.$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\EtP6E3lJgR.svg"> If <!-- $Q$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\cJcylPEtGz.svg"> is not one-to-one, the solution will not be unique. Then <!-- $Q^{-1}\left(B\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\eFqf8ha67i.svg"> defines a set of generalized contours on $\Lambda$, where each contour is the collection of <!-- $\lambda\ \in \Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\zSMVkqXbwX.svg"> such that <!-- $Q\left(\lambda\right)=b$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\EfalJph1eE.svg"> for some <!-- $b\in B.$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\uGUpdHEcmO.svg"> One consequence is that multiple distributions on the input
space will induce the observed distribution on the output space.

An estimated probability distribution <!-- $P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\bnqIynTx5o.svg"> on $\Lambda$ can be obtained as follows. First, for any <!-- $y\in D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\AYaJbYlcby.svg"> consider the generalized contour given by <!-- $C_y:=\{\lambda \in \Lambda : Q(\lambda) = y\}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\KAF2rdYgWZ.svg">. Then
the parameter space <!-- $\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\pjF9t50tCT.svg"> can be decomposed into the union of these generalized contours: 

$$\Lambda = \bigcup_{y\in D} C_y.$$

This produces a one-to-one map from <!-- $\{C_y\}_{y\in D}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\7wECGKWrdo.svg"> to <!-- $D.$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\syiWa4aHME.svg"> A solution to the SIP is found by making distributional assumptions on the generalized contours <!-- $\{C_y\}_{y\in D}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\qYHS5IWU7j.svg">, denoted <!-- $\{P_y\}_{y\in D}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\Q0tLXaPSqg.svg">. Combining the assumed distributions on the generalized contours with the observed output distribution <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\OWE94EaSwR.svg">, a solution to the SIP is obtained through disintegration:

$$P_\Lambda (A) = \int_{y\in D} \left(\int_{ \lambda\in C_y \cap A} dP_y(\lambda)\right) dP_D(y),\text{ for } A\in \mathcal F.$$

Essentially, to compute the probability of an event $A\in \mathcal F$,
you set a <!-- $y\in D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\mhnRNSUhiJ.svg"> and find the conditional probability of <!-- $A$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\XOYYqVDU03.svg"> given <!-- $C_y$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\Od3vfWQPE4.svg"> by integrating the assumed measure <!-- $P_y$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\TI1Mp8lo7E.svg"> on the generalized contour over all <!-- $\lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\ymak5dhvSw.svg"> contained in both <!-- $C_y$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\fXvjcLA1L7.svg"> and <!-- $A$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\ESMqc7jgww.svg">. Integrating
these conditional probabilities over all <!-- $y\in D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\IUkDf3f7Pk.svg"> with the observed measure <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\tnzAPfEz5e.svg"> produces <!-- $P_\Lambda (A)$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\0vEFXWhUEO.svg">.


## **Code Outline**

BET is a python-based package for solving stochastic forward and inverse problems. Documentation and installation instructions can be found [here](https://ut-chg.github.io/BET/overview.html#installation).
This repository explores inverse ensemble forecasting for COVID-19 through simulations and experimental data. We let <!-- $\tilde P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\z8RAdPpNdZ.svg"> be the probability distribution on the SIR's model parameters <!-- $\beta$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\Doj5Vq2ocV.svg"> (transmission rate) and <!-- $\gamma$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\bGCxUqmyvh.svg"> (recovery rate) on the sample space <!-- $\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\cE5vTCvAew.svg">, and use various QoI.

### Simulations

The general simulation steps are as follows:
- Choose data-generating distribution <!-- $\tilde P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\z8RAdPpNdZ.svg">. We use <!-- $\beta$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\Doj5Vq2ocV.svg"> ~ Beta(12, 30), <!-- $\gamma$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\bGCxUqmyvh.svg"> ~ Beta(6, 30) independently, shifted and scaled to the sample space $\tilde \Lambda = [0, 0.35]X[0,0.6] $.

- Generate “observed” data.
Draw parameter values from <!-- $\tilde P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\z8RAdPpNdZ.svg">. For each pair of paramter values drawn, solve the SIR and compute the QoI. This becomes the observed distribution <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\70SdivIgEg.svg"> on the QoI.
- “Forget” the data-generating distribution and use the SIP to compute a Bayesian solution to <!-- $\tilde P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\z8RAdPpNdZ.svg">. First choose the sample space <!-- $\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\cE5vTCvAew.svg">, then solve the SIP with the observed QoI on <!-- $\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\cE5vTCvAew.svg"> to obtain a distribution <!-- $P_\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\RVfnRrZOeE.svg"> consistent with <!-- $P_D$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\70SdivIgEg.svg">. 


#### Simulation1.py 
We use the QoI $Q_S(\lambda, T) = \frac{(S(T_0) - S(T_0 + T)}{T}$ (the additional infections that occurred within a time period $[T_0,  T_0 + T]$, scaled by $T$) and explore different choices of sample space <!-- $\Lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\cE5vTCvAew.svg"> on the solution. We use $T = 30$ and $T_0  = 10$.

#### Simulation2.py 
We use a new QoI  $Q_I(\lambda, T) = \frac{(I(T_0 + T) - I(T_0)}{T}$ (the difference in $\textit active$ infections between the two times) and explore different choices in observation time $T$. We use the sample space $[0, 0.35]X[0,0.6]$ and $T_0  = 10$.

#### Simulation3.py 
We use two-dimensional QoI by 1. combining information from both the susceptible and infectious populations at time $T$, and 2. using information from the susceptible populations at two time points. We use the sample space $[0, 0.35]X[0,0.6]$ and $T_0  = 10$, $T = 30$, $T_A  = 30$, and $T_B  = 31$.

<!-- $$
Q_{S,I}(\lambda, T) =  \begin{bmatrix}
    \frac{1}{T}\left(S_{\lambda}(T_0) - S_{\lambda}(T_0 + T)\right), &
    \frac{1}{T}\left(I_{\lambda}(T_0 + T) - I_{\lambda}(T_0)\right) 

\end{bmatrix}^{T},
$$ --> 

<div align="center"><img style="background: white;" src="svg\3EbRrz1f1U.svg"></div> 

<!-- $$
Q_{S, S}(\lambda, T_A, T_B) =  \begin{bmatrix}
    \frac{1}{T}\left(S_{\lambda}(T_0) - S_{\lambda}(T_0 + T_A)\right), & 
    
    \frac{1}{T}\left(S_{\lambda}(T_0) - S_{\lambda}(T_0 + T_B)\right)
\end{bmatrix}^{T}.
$$ --> 

<div align="center"><img style="background: white;" src="svg\KTc3ihgYSO.svg"></div>


#### Forecasting.py 
We take the solutions obtained from the simulation scripts and forecast future QoI.

### COVID-19 Data

We compute solutions during two COVID-19 surges using empirical data on county-level infection levels in the US, obtained from [The New York Times](https://github.com/nytimes/covid-19-data). From this data, we chose 113 counties containing college towns as our population of interest.

- Surge 1: September 20, 2020 - May 1, 2021. 
- Surge 2: December 1, 2021 - March 16, 2022.

Within each surge time frame, we determined start times $T_0$ for each county indiviually and collected the QoI at various time points $T$ beyond the start to use as the observed data in the inversions.

#### COVIDSurges.py 
We compute solutions on the parameter space using the observed data at various times in the two surges. We also use the observed data at later times to evaluate forecasting results from solutions obtained at earlier times.