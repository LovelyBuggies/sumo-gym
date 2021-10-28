import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sumo_gym",
    version="0.0.1",
    author="Shuo Liu",
    author_email="ninomyemail@gmail.com",
    description="OpenAI-gym like toolkit for developing and comparing reinforcement learning algorithms on SUMO.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lovelybuggies/sumo-gym",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=['gym'],
)