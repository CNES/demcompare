import nox


@nox.session
def lint(session):
    session.install('flake8',
                    'pep8-naming',
                    'flake8-mutable',
                    # 'flake8-eradicate',
                    'flake8-comprehensions',
                    'flake8-import-order'
                    )
    session.run('flake8', '-v', 'maj_mns_lib')
    session.run('flake8', '-v', '--ignore=D', 'tests')


# '3.10' version was removed temporary because Pytorch CPU is not compatible with this version yet
@nox.session(python=['3.8', '3.9'])
def tests(session):
    session.install('.[tests]')
    session.run('pytest', '-vv', '--cov-report', 'term-missing', '--cov=maj_mns_lib', 'tests/')
