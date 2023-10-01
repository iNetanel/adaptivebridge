echo "Basic PEP 8 aligment.."
autopep8 --in-place --recursive .

echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf adaptivebridge.egg-info
rm -rf __pycache__
rm -rf adaptivebridge/__pycache__
rm -rf adaptivebridge/.pylint.d
rm -rf .pytest_cache
rm -rf .pylint.d

echo "Buidling package..."
python setup.py sdist bdist_wheel
autopep8 --in-place --recursive .