mkdir ./ca-certs
cp /usr/local/share/ca-certificates/* ./ca-certs/.
docker build -t ${USER}/diffusers .
rm -fr ./ca-certs