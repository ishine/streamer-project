from setuptools import setup, find_packages


with open('tacotron/version.py') as f:
    for line in f.readlines():
        values = line.split('=')
        if 'version' == values[0].strip():
            version = eval(values[1])
            break

with open('requirements.txt') as f:
    deps = [str(dep.strip()) for dep in f.readlines() if not dep.startswith("--")]

setup(
    name='tacotron',
    packages=find_packages(exclude=["tests"]),
    version=version,
    install_requires=deps,
    description='Tacotron model',
    include_package_data=True,
    dependency_links = [
        'https://P4iP9-bsahyX98BawYhlQcHapiFsqvE@gem.neosapience.com/pypi/tts_text_util/',
        'https://P4iP9-bsahyX98BawYhlQcHapiFsqvE@gem.neosapience.com/pypi/voxa/'
    ]
)
