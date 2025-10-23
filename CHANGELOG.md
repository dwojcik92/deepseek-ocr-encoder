# CHANGELOG



## v1.0.0 (2025-10-23)

### Breaking

* feat: add configurable preprocessing hooks

This commit adds support for custom preprocessing transforms and exposes
resize/normalization parameters, allowing users to:

- Configure custom resize dimensions (including native resolution support)
- Customize normalization parameters (e.g., ImageNet vs CLIP)
- Inject custom preprocessing transforms for domain-specific pipelines
- Process pre-preprocessed tensors directly
- Fine-tune interpolation modes and quality settings

Key Features:
- preprocessing_transform: inject custom transform functions
- resize_size: configurable dimensions or None for native resolution
- resize_interpolation: BICUBIC, LANCZOS, BILINEAR, etc.
- normalization_mean/std: custom normalization parameters
- skip_default_preprocessing: accept pre-processed tensors

This enables reusing existing preprocessing pipelines without forking
the codebase while maintaining full backward compatibility.

BREAKING CHANGE: None - all changes are opt-in with backward compatible defaults ([`b3dcb37`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/b3dcb3745f9f269f8377f767696f5b41ea5208b7))

### Fix

* fix: enhance document preprocessing with sharpening filter for better text recognition ([`005875d`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/005875dc254217988dc3fd4515d882f91c0081cc))

* fix: set default value for resize_size in DeepSeekOCREncoder ([`8e4eb7e`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/8e4eb7e7909b1974e362c08c991a733b41e67ff8))

* fix: enable method chaining for mock model in tests ([`8c85448`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/8c85448d12ad5b8b7fe1ca91b1b0c1ce64e236fb))

* fix: update mock position embeddings to use a detachable tensor ([`e295743`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/e2957437d6eb04946b5ffba76d14b43ba1a0c58f))

### Unknown

* Merge pull request #9 from dwojcik92/feat/configurable-preprocessing-hooks

feat: add configurable preprocessing hooks for custom image transformations ([`79ed4b4`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/79ed4b4a0ea3d15c0b82ed121dd2a4d1775d409e))

* Update tests/test_custom_preprocessing.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`7a729b2`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/7a729b2b318a674472bd12cc24034c4e419f0905))

* Update tests/test_encoder.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`b6f7eee`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/b6f7eee7181ad7fc8a19840fa6c7c6dd78d544b1))


## v0.2.1 (2025-10-23)

### Fix

* fix: clone CUDA graph output buffer to prevent data corruption

When CUDA graph is captured, successive encode calls would return the same
tensor buffer, causing silent data corruption when storing multiple results.
Now returns a clone of the static output buffer. ([`5be369f`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/5be369fd1068a756c265568457dcfc662de79bc9))


## v0.0.2 (2025-10-23)

### Fix

* fix: clone output tensor to prevent buffer reuse in encoder ([`5b4285b`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/5b4285b8a663b9a120da28d92a4caa6ead364971))


## v0.0.1 (2025-10-23)


## v0.2.0 (2025-10-23)

### Ci

* ci: fix build module installation in semantic release workflow ([`83a8f2f`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/83a8f2f654c8355b1f6cba35437d1f2e33880497))

* ci: replace manual publish workflow with automated semantic release

- Remove python-publish.yml (manual release workflow)
- Keep release.yml for automated semantic versioning and publishing
- Workflow triggers on push to main branch
- Automatically handles version bumps based on conventional commits ([`68c6f39`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/68c6f39be3bf98d462609dbd5cdfae3369c24dfb))

### Documentation

* docs: add GitHub Copilot instructions for project setup and commit conventions ([`4628c3a`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/4628c3a90206286ab1822fcfac66c0311d48b087))

### Fix

* fix: update build command and add missing dependencies in pyproject.toml ([`9d5e7f6`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/9d5e7f6826259070e06cdac2ceb9d6819a688daf))

* fix: add missing addict dependency required by DeepSeek-OCR model

The addict package is required by the DeepSeek-OCR model&#39;s remote code.
Without it, users encounter an ImportError when calling from_pretrained().
This fix adds addict&gt;=2.4.0 to the package dependencies. ([`361edba`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/361edbaf684aab4eba30e149aae4f97d87c67f55))

### Unknown

* Update PyPI project URL in workflow configuration ([`afcb1c4`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/afcb1c4fa43dd6a4a33e18c58b599ff7d4cfb712))

* Refactor installation instructions in README.md for clarity and consistency ([`0e777aa`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/0e777aad8b446a5c740245b3c6dca61708d67802))

* Merge branch &#39;main&#39; of github.com:dwojcik92/deepseek-ocr-encoder ([`9617e49`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/9617e49bcb92b460157441d2b71d334d17f60cea))

* bump version to 0.2.0 and update Python requirement to &gt;=3.12; adjust torch dependency to &gt;=2.4.0 ([`cc64474`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/cc64474ab1d510cc484b468d094fd37adb882306))

* Merge pull request #8 from dwojcik92/copilot/fix-importerror-llamaflashattention2

[WIP] Fix ImportError for LlamaFlashAttention2 in dependencies ([`171ef93`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/171ef93c8cc031ff5e60f0e526eec2e28e39f7c6))

* Merge branch &#39;main&#39; into copilot/fix-importerror-llamaflashattention2 ([`a6ea251`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/a6ea25135ee583e8ee016a70e590510da00ae8a2))

* Merge branches &#39;main&#39; and &#39;main&#39; of github.com:dwojcik92/deepseek-ocr-encoder ([`fcc2af3`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/fcc2af32b7621c00cda730521d06401630900fb3))

* Merge pull request #2 from dwojcik92/copilot/add-pdf-multi-page-support

Add PDF multi-page support with PyMuPDF integration ([`1481d4a`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/1481d4a8cdebaa722dd6709b37b3ff9c53fc2235))

* Merge branch &#39;main&#39; into copilot/add-pdf-multi-page-support ([`a86f3f3`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/a86f3f36b16bc5eab3ee3ca11e9f53eb5b3395c9))

* Pin transformers version and add compatibility checks

Co-authored-by: dwojcik92 &lt;10101471+dwojcik92@users.noreply.github.com&gt; ([`3a2870a`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/3a2870a33515ef7c422016dd22da2c37cd36fbcc))

* Merge pull request #6 from dwojcik92/copilot/update-readme-deepseek-ocr

Update README with citation and info about DeepSeek-OCR paper (arXiv:2510.18234v1) ([`7ea8c64`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/7ea8c647fb87a7f7749d7f29a113e07c19ba221a))

* Merge branch &#39;main&#39; of github.com:dwojcik92/deepseek-ocr-encoder ([`4cce0b5`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/4cce0b54553c327b2f6d95150d40f5ff4c09e7b6))

* updated readme ([`9fa444a`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/9fa444a29aab718d4e6b5b8b75cf0dd37974d4a4))

* Initial plan ([`8f29546`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/8f295466506d47e1ab5c9a73c404e503b66812a1))

* Merge from_pretrained() method from main branch - resolve conflicts

Co-authored-by: dwojcik92 &lt;10101471+dwojcik92@users.noreply.github.com&gt; ([`fc3f6a3`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/fc3f6a325db3e360f363454aeecc370f0457d55c))

* Update README with DeepSeek-OCR paper citation and resources

Co-authored-by: dwojcik92 &lt;10101471+dwojcik92@users.noreply.github.com&gt; ([`6e5f84b`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/6e5f84b0eb2667acfa66c92e442f307eb0c5de15))

* Initial plan ([`11ccbed`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/11ccbedab3be691400b3fbe942b7e337a8d308ce))

* Remove unnecessary touch() call in test and add clarifying comment

Co-authored-by: dwojcik92 &lt;10101471+dwojcik92@users.noreply.github.com&gt; ([`469390e`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/469390e51c8f39e5e846deff44937984a359199d))

* Merge pull request #4 from dwojcik92/copilot/add-simplified-model-loader

Add simplified from_pretrained() API for one-line model initialization ([`ea93649`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/ea936495c2bb42ec236a75405ccdf6acd87caa4f))

* Update tests/test_encoder.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`b359c38`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/b359c38f1c89820c486a84ef01fbb2e7c3b38713))

* Update tests/test_encoder.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`3f60ba2`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/3f60ba20c3ecf86e9329c9c14732ab30746f1fd9))

* Update src/deepseek_ocr_encoder/encoder.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`ddfa1af`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/ddfa1af166fbaf547c4cb4482b1f18dc130ce3ea))

* Update tests/test_encoder.py

Drop is True

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`157ff89`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/157ff89aaedfa8eb399a5862671331cd98819e86))

* Add from_pretrained() class method for simplified model loading

Co-authored-by: dwojcik92 &lt;10101471+dwojcik92@users.noreply.github.com&gt; ([`1268281`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/12682817519117cf495dd1e6f440295294765e1a))

* Initial plan ([`9bcb086`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/9bcb08691c63c518c92c902191009db626ef9fad))

* Add PDF (multi-page) support with PyMuPDF integration

Co-authored-by: dwojcik92 &lt;10101471+dwojcik92@users.noreply.github.com&gt; ([`6ee40d1`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/6ee40d12350ec6be027403966c9f0cb45d7a2397))

* Initial plan ([`1edf836`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/1edf836fcd2733f5151e463dbe5f3ad7e20b59a6))

* Update author information in pyproject.toml ([`af679e2`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/af679e2023ccbc1083ff7315c802b940345ad234))

* Cleaning repo. ([`0d1889e`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/0d1889ebeb6c946c68b2b64cbe8725d5d8ce1518))

* Add DeepSeek OCR Encoder package with optimized image encoding

- Implemented DeepSeekOCREncoder class for efficient vision token generation.
- Created README.md and SETUP.md for installation and usage instructions.
- Added example script for basic usage of the encoder.
- Established project structure with pyproject.toml for dependencies and configurations.
- Included unit tests for encoder functionality and initialization.
- Integrated CUDA graph support for optimized inference.
- Ensured memory efficiency by removing unused model components. ([`36d328a`](https://github.com/dwojcik92/deepseek-ocr-encoder/commit/36d328a6ca475d8dae7fe30fc7d565a68e207358))
