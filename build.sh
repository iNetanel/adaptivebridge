
echo "AdaptiveBridge source packadge builder:"

echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf adaptivebridge.egg-info
rm -rf __pycache__
rm -rf adaptivebridge/__pycache__
rm -rf adaptivebridge/.pylint.d
rm -rf .pytest_cache
rm -rf .pylint.d
rm requirements.txt

echo "Run components tests..."
pytest --cov=adaptivebridge

echo "Basic PEP 8 aligment.."
autopep8 --in-place --recursive .

echo "Create requirements file..."
pipreqs .

echo "Buidling package..."
python setup.py sdist bdist_wheel

echo "Packadge check..."
twine check dist/*