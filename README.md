# So-DAS
**So-DAS: A Two-Step Soft-Direction-Aware Speech Separation Framework**

**This work has been accepted by *IEEE Signal Processing Letters (SPL).*  Early access available [here][Paper].**

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


[Paper]: https://ieeexplore.ieee.org/abstract/document/10052748
[sms_wsj]: https://github.com/fgnt/sms_wsj
[asteroid]: https://github.com/asteroid-team/asteroid
