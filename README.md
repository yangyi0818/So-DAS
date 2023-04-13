# So-DAS
**So-DAS: A Two-Step Soft-Direction-Aware Speech Separation Framework**

**This work has been published on *IEEE Signal Processing Letters (SPL).*  The paper is available [here][Paper].**

## Contents 
* **[So-DAS-A-Two-Step-Soft-Direction-Aware-Speech-Separation-Framework](#so-das-a-two-step-soft-direction-aware-speech-separation-framework)**
  * **[Contents](#contents)**
  * **[Introduction](#introduction)**
  * **[Dataset](#dataset)**
  * **[Requirement](#requirement)**
  * **[Train](#train)**
  * **[Test](#test)**
  * **[Results](#results)**
  * **[Citation](#citation)**
  * **[References](#references)**

## Introduction
**Most existing direction-aware speech separation systems lead to performance degradation when the angle difference between speakers is small due to the low spatial discrimination. To address this issue, we propose a two-step soft-direction-aware speech separation (So-DAS) framework, which consists of a direction of arrival (DOA) estimation module and a speech separation module. First, the two modules are individually optimized, and directional features (DFs) derived from ground-truth DOAs are utilized as spatial information to facilitate the separation module. Next, the two modules are cascaded and optimized with only separation loss, and the DFs are generated using the estimator outputs. By this means, the consistency between the two modules is strengthened, and thus spatial cues that are more beneficial to the separation task can be exploited by the network itself. The experimental results show that compared to the baselines, DFs extracted by our proposed method provides clearer superiority, especially when the angle difference between speakers is small. In addition, our approach yields a state-of-the-art word error rate of 3.4% on the real-recorded utterance-wise LibriCSS dataset.**

![image](https://github.com/yangyi0818/So-DAS/blob/main/figures/framework.png)

## Dataset
**We use [sms_wsj][sms_wsj] to generate room impulse responses (RIRs) set. ```sms_wsj/reverb/scenario.py``` and ```sms_wsj/database/create_rirs.py``` should be replaced by scripts in 'sms_wsj_replace' folder.**

**use ```python generate_rir.py``` to generate training and valadation data**

## Requirement
**Our script use [asteroid][asteroid] toolkit as the basic environment.**

## Train

**```./run.sh --id 0,1,2,3```**

**Step 1: Optimize the DOA estimator and the separator individually.**

**Step 2: Cascade the two modules as a whole system and train it with only separation loss.**

## Results

![image](https://github.com/yangyi0818/So-DAS/blob/main/figures/table1.png)
![image](https://github.com/yangyi0818/So-DAS/blob/main/figures/table2.png)

## Citation
**Cite our paper by:** 

**@ARTICLE{10052748,**

  **author={Yang, Yi and Hu, Qi and Zhao, Qingwei and Zhang, Pengyuan},**
  
  **journal={IEEE Signal Processing Letters},**
  
  **title={So-DAS: A Two-Step Soft-Direction-Aware Speech Separation Framework},**
  
  **year={2023},**
  
  **volume={30},**
  
  **pages={344-348},**
  
  **doi={10.1109/LSP.2023.3248952}**
  
**}**

## References

**[24] F. Dang, H. Chen, and P. Zhang, “DPT-FSNet: Dual-path transformer based full-band and sub-band fusion network for speech enhancement,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2022, pp. 6857-6861.**

**[26] Z. Chen, T. Yoshioka, L. Lu, T. Zhou, Z. Meng, Y. Luo, J. Wu, X. Xiao, and J. Li, “Continuous speech separation: Dataset and analysis,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2020, pp. 7284-7288.**

**[32] Z.-Q. Wang, P. Wang, and D. Wang, “Multi-microphone complex spectral mapping for utterance-wise and continuous speech separation,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 29, pp. 2001- 2014, 2021.**

**[40] S. Chen, Y. Wu, Z. Chen, J. Wu, J. Li, T. Yoshioka, C. Wang, S. Liu, and M. Zhou, “Continuous speech separation with conformer,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2021, pp. 5749-5753.**

**[41] K. Saijo and R. Scheibler, “Spatial loss for unsupervised multi-channel source separation,” in Proc. Interspeech, 2022, pp. 241-245.**

**Please feel free to contact us if you have any questions.**
  
[Paper]: https://ieeexplore.ieee.org/abstract/document/10052748
[sms_wsj]: https://github.com/fgnt/sms_wsj
[asteroid]: https://github.com/asteroid-team/asteroid
