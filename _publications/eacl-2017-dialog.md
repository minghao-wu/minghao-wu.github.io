---
title: "Dialog state tracking, a machine reading approach using Memory Network"
collection: publications
permalink: /publication/eacl-2017-dialog
date: 2017-04-05
venue: 'Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)'
paperurl: 'http://www.aclweb.org/anthology/E17-1029'
paperurltext: 'Link to ACL anthology'
citation: 'Julien Perez and <b>Fei Liu</b> (2017) <a href="http://liufly.github.io/files/papers/eacl-2017-dialog.pdf"><u>Dialog state tracking, a machine reading approach using Memory Network</u></a>, In <i>Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)</i>, Valencia, Spain, pp. 305-314.'
---

```
@InProceedings{Perez+:2017,
  author    = {Perez, Julien  and  Liu, Fei},
  title     = {Dialog state tracking, a machine reading approach using Memory Network},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)},
  year      = {2017},
  address   = {Valencia, Spain},
  pages     = {305--314}
}
```

## Abstract
In an end-to-end dialog system, the aim of dialog state tracking is to accurately estimate a compact representation of the current dialog status from a sequence of noisy observations produced by the speech recognition and the natural language understanding modules. This paper introduces a novel method of dialog state tracking based on the general paradigm of machine reading and proposes to solve it using an End-to-End Memory Network, MemN2N, a memory-enhanced neural network architecture. We evaluate the proposed approach on the second Dialog State Tracking Challenge (DSTC-2) dataset. The corpus has been converted for the occasion in order to frame the hidden state variable inference as a question-answering task based on a sequence of utterances extracted from a dialog. We show that the proposed tracker gives encouraging results. Then, we propose to extend the DSTC-2 dataset with specific reasoning capabilities requirement like counting, list maintenance, yes-no question answering and indefinite knowledge management. Finally, we present encouraging results using our proposed MemN2N based tracking model.