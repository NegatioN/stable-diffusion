curl micro.mamba.pm/install.sh | bash
# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE="/root/.local/bin/micromamba";
export MAMBA_ROOT_PREFIX="/root/micromamba";
__mamba_setup="$('/root/.local/bin/micromamba' shell hook --shell bash --prefix '/root/micromamba' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    if [ -f "/root/micromamba/etc/profile.d/micromamba.sh" ]; then
        . "/root/micromamba/etc/profile.d/micromamba.sh"
    else
        export  PATH="/root/micromamba/bin:$PATH"  # extra space after export prevents interference from conda init
    fi
fi
unset __mamba_setup
# <<< mamba initialize <<<
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
micromamba create -f environment.yaml -y
