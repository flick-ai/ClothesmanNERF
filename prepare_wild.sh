SUBJECT="jntm"

cd ./tools/prepare_wild
python prepare_dataset.py --subject ${SUBJECT}
cp -r /home/wenhao/CV/Dataset/${SUBJECT} ../../dataset/wild/${SUBJECT}