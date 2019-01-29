# Install build dependencies
sudo apt-get update
sudo apt-get install unzip
sudo apt-get install -y build-essential libbz2-dev libssl-dev libreadline-dev \
     libffi-dev libsqlite3-dev tk-dev
sudo apt-get install -y libpng-dev libfreetype6-dev

# Install pyenv
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# pyenv install the right python version
pyenv install 3.7.0
pyenv global 3.7.0

exec $SHELL
