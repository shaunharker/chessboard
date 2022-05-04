# installer script

git submodule update --init --recursive
pip install . --ignore-installed --no-cache-dir --use-feature=in-tree-build
