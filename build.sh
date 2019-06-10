#!/bin/bash
set -e -x
PLAT=manylinux2010_x86_64
# this script should be executed within lemon:v2 docker
# Compile wheels
PY36=/opt/python/cp36-cp36m/bin
PY37=/opt/python/cp37-cp37m/bin
$PY36/pip wheel . -w dist/
$PY37/pip wheel . -w dist/
# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w dist/wheelhouse/
done

