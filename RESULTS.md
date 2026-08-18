# Results and comparison

## Main result

The final model achieved **0.603 ± 0.014 macro subject CER** on 440 held-out sentences from 19 subjects. The comparable Spanish asynchronous result reported in Brain2Qwerty v2 is **0.59 ± 0.02**.

Our result is 0.013 CER, or 1.3 percentage points, above the paper mean and lies inside its reported 0.57–0.61 range.

## Comparison

| System | Dataset | Timing supplied at inference? | Decoder | CER | Direct comparison? |
|---|---|---:|---|---:|---|
| Initial local run | SpanishBCBL | No | Greedy CTC | 0.697 | Yes, but training scale was incorrect |
| **Final reproduction** | **SpanishBCBL** | **No** | **Greedy CTC** | **0.603 ± 0.014** | **Yes** |
| Paper Encoder Async | SpanishBCBL | No | Greedy CTC | 0.59 ± 0.02 | Yes |
| Paper Encoder Sync | SpanishBCBL | Yes, keystroke timestamps | Character classifier | 0.39 ± 0.02 | No—easier timestamp-assisted task |
| Paper Encoder Async | EnglishBCBL | No | Greedy CTC | 0.25 ± 0.03 | No—approximately 10× more recording per subject and more varied sentences |
| Paper Encoder + 6-gram | EnglishBCBL | No | CTC + character language model | 0.26 ± 0.03 | No—different dataset |
| Full Brain2Qwerty v2 | EnglishBCBL | No | Encoder + aligner + LLM | 0.31 ± 0.03 | No—different dataset and optimized for words/meaning |

Source: [Brain2Qwerty v2 paper](https://facebookresearch.github.io/brain2qwerty/assets/brain2qwerty_v2.pdf).

## Improvement over the initial run

- Initial CER: 0.697
- Final CER: 0.603
- Absolute reduction: **0.094**, or **9.4 percentage points**
- Relative error reduction: **13.5%**

The improvement came from matching the paper's effective batch size of 1,024 and scaling the 500-step learning-rate warmup correctly. The initial run used an effective batch of 64, exposing the model to approximately 16 times fewer samples during warmup.

## Training and generalization

| Measurement | Result |
|---|---:|
| Parameters | 352,449,614 |
| Best validation CER | 0.574 |
| Best epoch | 96 |
| Early-stop epoch | 146 |
| Final macro subject test CER | 0.603 ± 0.014 |
| Global micro test CER | 0.611 |
| Validation-to-test gap | 0.029 |
| Per-subject CER range | 0.469–0.680 |
| Held-out subjects represented | 19 |
| Held-out sentences | 440 |

Macro subject CER is the headline number because it matches the paper: CER is calculated per sentence, averaged within each subject, and then averaged across subjects.

## Assessment

**As a reproduction, the training was successful.** It reproduced the paper within its reported uncertainty and improved substantially over the incorrectly scaled initial run. The validation-to-test gap is moderate, and the checkpoint selection worked as intended.

**As a transcription model, performance remains limited.** A CER near 0.60 means the raw greedy output still contains many character errors. This reflects the difficult asynchronous task and the small, repetitive SpanishBCBL corpus rather than a failed training run. The model is appropriate as a research reproduction and baseline, not as a high-accuracy typing system.

## Report-ready summary

> We trained a 352-million-parameter convolutional Conformer with a CTC objective to decode complete Spanish sentences directly from continuous 306-channel MEG recordings without keystroke timestamps. On 440 held-out sentences from 19 subjects, the model achieved a macro subject CER of 0.603 ± 0.014. This is within the uncertainty range of the 0.59 ± 0.02 Spanish asynchronous encoder reported by Brain2Qwerty v2. Correcting the effective batch size and learning-rate warmup reduced CER from 0.697 to 0.603, a 13.5% relative error reduction. The experiment therefore constitutes a successful reproduction, although the remaining error rate is still too high for reliable transcription without improved decoding, additional diverse training data, or subject adaptation.
