"""Built-in memory corpus + the engines that read and curate it.

The corpus contains task histories, cross-task patterns, and the scalar task_categories.json mapping. memory/store.py serves both modes through a BundleMemoryStore: source builds its launch snapshot from the corpus root, while the frozen binary imports the generated bundle.
"""
