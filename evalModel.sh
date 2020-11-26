model=$1
shift
uname=$1
shift
psd=$1
shift
echo "
"
echo "$@"
python3 -m trainers.$model "$@" <<<"$uname
$psd"
exit 0
