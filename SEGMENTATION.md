# Automatic masking

Regenerate every image in the book selected in `config.yaml`, without previews:

```powershell
.venv\Scripts\python.exe main.py --all
```

This overrides the configured page selection and replaces that book's processed
images. Without `--all`, the configured pages and preview settings apply.

Output filenames include the English bird name, for example
`352-Brent-Goose.jpg`. Additional instances on a page include an instance
suffix, such as `53-v1-White-bellied-Swift.jpg`, to avoid overwriting each other.

Segmentation defaults to `device: auto`: CUDA is preferred when ONNX Runtime
GPU is installed, with a CPU fallback if GPU inference fails. The active device
is logged. Set `device: cpu` to force CPU or `device: cuda` to require CUDA.
To add GPU dependencies after installing the base requirements:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

The NVIDIA runtime DLLs are preloaded from the virtual environment. Reinstalling
the CPU ONNX package can replace the shared runtime files; reinstall the GPU
package afterwards if CUDA disappears. On 4 GB GPUs this model can approach
the memory limit, so GPU execution does not guarantee faster batch processing.
The original full-precision model and 1024-pixel input are retained.

Orientation uses local OCR on the visible scan in four
rotations and looks for a confident horizontal caption matching the bird's
English label. OCR's automatic upside-down text correction is disabled for
this check. Embedded PDF OCR is not used: garbled text and block geometry can
misrepresent sideways captions. Without a matching caption, the scan orientation
is retained.

The configured pipeline first checks the oriented scan for text with local OCR.
Subject detection excludes long scan borders and narrow broken border fragments
in the outermost page margins, preserving similar shapes inside the page.
It detects illustration outlines after that narrow text cleanup, before color
correction, smoothing, or brightness adjustment. The final mask is applied to
the corrected color image. The old brightness-subtraction mask and manual
removal step are no longer used by `config.yaml`.

`AutomaticTextCleanupStep` uses the bundled RapidOCR ONNX models locally. It
requires recognized text with at least four letters, confidence >= 0.8, and a
small horizontal region surrounded by relatively quiet texture. Conventional
signature endings (`del`, `lith`, `sculp`, `pinx`) allow confidence >= 0.75 with
the same texture checks. Dark strokes inside the recognized polygon are
inpainted, preserving the surrounding paper/ground color rather than whitening
the whole rectangle. Uncertain or textured regions are retained. These checks
reduce false removals; they cannot prove that text never overlaps artwork.
Per-plate detection and removal reports are saved under `data/out/text-cleanup/`.

`PaperBalanceStep` estimates paper color from the brighter background pixels
outside the subject mask, excluding outer scan shadows. It neutralizes the
paper color cast and compensates for slowly varying illumination with a bounded
shift (10 brightness levels maximum, strength 0.5). Brightening is capped at
1.1x and highlights use a smooth shoulder instead of abrupt clipping. Without
reliable paper samples, the original tone is preserved. No pixel on the bird
is whitened merely because it resembles paper.

The final crop blends inward along an eight-pixel boundary after brightness
adjustment. This softens hard mask edges without reintroducing paper outside
the selected illustration. Birds and illustrated scenery remain the target.
The final brightness/contrast settings are -5/15 with a smooth highlight
shoulder during cleanup. Once wash removal and cropping finish,
`FinalColorEnhancementStep` is the last pipeline step. The configured
`detail_preserving` mode grades float Lab luminance with a smooth monotonic tone
curve, deepening shadows while retaining differences between pale feather
shades. Adaptive vibrance (0.25) boosts muted colors mainly in the midtones,
tapering off in highlights and already saturated colors. White padding and
cleared wash remain pure white. Background removal remains the responsibility
of the preceding masks and wash cleanup; grading does not classify pale pixels
as background.

The previous vibrant boost is still available with `mode: legacy`,
`reference_image: oriented_image`, strength 1.0, correction bounds 0.8–1.2,
brightness -40, and contrast 30. It clips pale feather shading, so it is no
longer the configured default.

`FadePaintedWashStep` reduces light, low-texture gray/blue wash while retaining
scenery. BiRefNet General Lite runs on the original scan at threshold 0.35 to
provide a protective foreground mask. This mask is **not** used to cut away
scenery. The wash step additionally protects strong strokes, warm earth tones,
and enclosed neutral uncertain areas inside that protective mask. Smooth blue
wash may be removed from added safety margins and filled gaps between reeds,
while the original learned foreground remains protected. Small blue gaps can
qualify independently of the broad-wash area threshold. Exposed wash and
large open gaps between scenery edges can fade toward white; dark, saturated,
and detailed areas are retained. Processing uses a 1600-pixel reference size,
then applies a soft fade at full scan resolution after contrast adjustment.
Open wash interiors far from dark scenery strokes are replaced fully with white.
Near water strokes and other dark detail, the gentler texture-dependent fade is
retained. This reduces faint wash speckles without opening white gaps in scenery.
Small grain holes and pale boundary fragments are bridged within those open
wash regions. Edge protection requires support from darker ink, rather than
protecting every faint boundary of the wash itself. Learned subject protection,
warm colors, and nearby dark scenery still prevent stronger whitening.

The default pipeline now needs the local BiRefNet model described below. Nothing
is sent to a remote service. Its learned mask can still miss unusual subjects,
so validation across the collection remains necessary. This targets cool/neutral
wash conservatively, rather than guaranteeing every painted background is gone.

Detection uses a grayscale preview with a maximum side of 1600 pixels. It finds
edges, closes small gaps, fills contour interiors, removes thin scan-border
components, and retains substantial disconnected components. A small expansion
protects fine outlines. The binary mask is then resized to the scan dimensions
with nearest-neighbor interpolation. If no usable contours are found, the scan
is preserved rather than erased.

These settings apply to every plate; there are no book/page exceptions. Fixed
detection resolution makes the gap and edge settings comparable across scans.
Multiple subjects are retained instead of selecting only the largest bird.

This mask separates illustration detail from plain paper. It does not identify
birds semantically: touching plants, water, painted sky, and some lettering can
remain. Broken or extremely faint outlines can still cause missing detail.
Perfect bird-only masks across varied historical artwork would require a model
validated on labeled examples from these books. Color matching alone cannot
reliably separate white plumage from similarly colored paper.

## Validation

Run from the project root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/download_models.py
.venv/Scripts/python.exe scripts/review_segmentation.py
```

The review processes 12 selected plates across all five volumes and writes new
images, coverage metrics, and a before/after contact sheet under
`data/out/segmentation-review/`. It does not replace processed originals.
The tests cover pale interiors, multiple subjects, scan borders, resolution
consistency, hole filling at image boundaries, and empty-detection fallback.

For future tuning, expand the review set and inspect masks alongside original
scans. Coverage is a useful diagnostic, not proof of correct segmentation. Before
claiming correctness for the entire collection, compare masks against a labeled
validation set including white feathers, fine legs, separate birds, uneven paper,
and overlapping scenery. Prefer retaining uncertain detail over erasing it.

## Learned segmentation

`SemanticIllustrationMaskStep` adds local ONNX inference on the oriented original
color scan. It accepts NCHW float32 RGB inputs and a first output shaped
`[1, classes, height, width]`. Normalization, input size, output logits/probabilities,
foreground class IDs, and mask threshold are configurable. The model session is
reused across plates. No image is uploaded and no model is downloaded at runtime.

Install the optional runtime and run the experimental comparison:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-semantic.txt
.venv/Scripts/python.exe main.py --config config.semantic.yaml --no-preview
.venv/Scripts/python.exe scripts/review_segmentation.py --config config.semantic.yaml --output data/out/semantic-review
```

The candidate model is BiRefNet General Lite, downloaded from the
[rembg model release](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx)
to `data/models/birefnet-general-lite.onnx`. Its MD5 was checked against the
[upstream model adapter](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/birefnet_general_lite.py):
`4fab47adc4ff364be1713e97b7e66334`. Preprocessing uses ImageNet RGB normalization
and 1024x1024 input as specified by the
[upstream inference implementation](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/birefnet_general.py).
Unlike the upstream adapter, inference does not rescale every prediction to
span zero to one: that could turn a uniformly weak prediction into foreground.

**This candidate is foreground/background segmentation, not a trained print-artifact
classifier.** On swan 355 it removes the background effectively but also drops
intentional scenery. That direct foreground cutout is not the production default;
the production wash cleanup uses it only as a protective mask. Original processed
images remain unchanged by the semantic comparison.

For the desired output, train or fine-tune a segmentation model on masks from
these books. Label paper, bird, scenery, lettering, and print damage separately,
or use a binary artwork/artifact mask. For a five-class output ordered that way,
set `foreground_classes: [1, 2]` to retain bird and scenery. Ensure the exported
model's input normalization and class mapping match the configuration.
Split validation by plate/book rather than random patches from the same scan,
and measure both retained artwork and remaining artifact pixels. Include faint
feathers, stems, uneven paper, marginal text, and artwork overlapping stains.

Segmentation cannot recover artwork obscured by damage. Damage within retained
artwork requires restoration/inpainting, with its own validation for altered
details. Zero artifacts on every plate is not guaranteed by a pretrained model.
Near-empty or near-full semantic masks fail with the plate identity instead of
silently falling back to an unclean scan. Coverage checks catch gross failures,
not loss of scenery; visual or labeled-mask validation is still required.
