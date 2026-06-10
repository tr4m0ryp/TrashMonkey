# What's Happening -- Plain-Language Summary

Companion read for `research/yolo11-waste-detection-finetune.md` (the full
design doc with all evidence and citations). This file just explains it.

## The goal in one paragraph

We're teaching a small, fast object-detection model (YOLO11 nano) to recognize
six kinds of waste -- plastic, paper, cardboard, metal, glass, organic -- as
single items sliding past on a white conveyor background. It runs on a Jetson
Orin Nano mini-computer watching three cheap ESP32 wifi cameras. If the model
is confident, the machine sorts the item into the right bin; if not, the item
just rides off the end into a "rest" bin. We never train a "rest" class -- rest
simply means "nothing was confident enough in time".

## The plan, step by step

**1. Get training images for free.**
We don't photograph anything ourselves. Public waste datasets (TrashNet and
friends) already contain thousands of photos of trash on clean backgrounds --
close enough to our white conveyor. We download several of them and translate
their labels into our six classes. (Which datasets exactly, and the full
translation table, is the one piece still being researched right now.)

**2. Draw boxes automatically.**
Most public sets only say "this photo is plastic" -- they don't say where the
object is, and a detector needs a box around it. We use an AI labeler
(Grounding DINO): you give it a text prompt like "plastic bottle" and it draws
the box. The class name comes from the dataset folder, so the labeler can't
mislabel anything -- it only locates. Worst case (white paper on a white
background, where simpler methods provably fail) gets a backup method, and a
10% human spot-check makes sure the boxes are good enough. Research says boxes
can be ~10-20% sloppy before training noticeably suffers, so this is safe.

**3. Train the model.**
Standard fine-tuning of a pretrained model, with every setting pinned down and
seeded so results are reproducible (the defaults silently change behavior with
dataset size -- we found and avoided that trap). The clever part is the
augmentation: our cheap cameras produce blurry, noisy, oddly-colored JPEG
images, and those defects are well documented. So during training we
deliberately degrade the nice training photos the same way -- compression
artifacts, sensor noise, motion blur, color drift -- and the model learns to
not care. Training runs on Google Colab through a manager notebook (the
ESPResso-V2 pattern: the notebook only orchestrates, all real code lives in
the package).

**4. Test honestly.**
A model always looks great on photos from the same datasets it trained on, so
we keep three scores: a normal validation score (the optimistic one), a score
on an entire dataset the model never saw (how well it generalizes), and the
key one -- that same unseen set degraded to look like our cameras. The third
number is our best prediction of demo-day performance. If the model scores
below 95% on validation we first fix data, and only then move up to the
slightly larger "small" model -- evidence says this task is easy enough that
nano should pass.

**5. Decide bins with votes, not one guess.**
Each item is seen in roughly 5-15 frames as it crosses the camera zone.
Instead of trusting one frame, frames vote: an item goes to a bin only if at
least 3 frames agree on the same class, that class wins the vote, and at least
one frame was quite confident. Anything else falls to rest. This matters
because a single confidence check is provably leaky -- unknown objects (stuff
that's none of the six classes) sneak past it 70-80% of the time, while a
voting rule needs them to fool the model the same way repeatedly. The vote
thresholds are tuned on degraded validation images, including a probe set of
deliberately-unknown objects built from the categories we dropped in step 1.

**6. Put it on the Jetson.**
The trained model is converted on the Jetson itself (the converted engine only
works on the machine that built it) into a half-precision TensorRT engine. It
ends up absurdly fast: ~5 ms per frame, while the cameras can only deliver
~8 frames a second -- the cheap cameras are the bottleneck, not the AI. One
small Python program reads the three camera streams, always keeps only the
freshest frame, runs them round-robin through the model, counts the votes, and
tells the sorting hardware: class, confidence, timestamp. Whole decision
typically takes ~0.1-0.2 s of a 1-2 s window. Comfortable.

## Decisions already made for you (review in /refine)

Ten preference calls were made to avoid stopping the run -- the optimizer
recipe, the auto-labeler choice, the voting numbers, JetPack version, FP16 vs
INT8, Colab training, and so on. Each is listed in the design doc under
"Decisions Made For You" with the alternative and when you'd want to flip it.

## What's NOT decided yet

One research agent is still finishing the dataset census: exactly which public
datasets we use, the label translation table, how many images per class, how
to deduplicate overlapping sets, and the license table for the report. That
fills the last four decisions (T1, T2, T4, T10) and unblocks the build of the
data pipeline. Everything else is ready.

## What happens next

1. Census lands -> last four decisions get filled in -> research phase done.
2. `/flow:refine` -- you read and adjust the decisions (optional).
3. `/flow:readyforlaunch` -- the design gets decomposed into build tasks and
   implemented: data pipeline first, then training, evaluation, threshold
   tuning, and the Jetson runtime, with the paper updated along the way.
