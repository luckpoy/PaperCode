case $1 in
    train_synth)
        python main.py train_synth 
        ;;
    tc)
        python main.py train_ctw
        ;;
    tec)
        python main.py test_ctw --model /home/ubuntu/bo/CCD/model/ctw/model.pkl
        ;;
    tesc)
        python main.py test_proj --model ./model/uname/model.pkl
        ;;
    train_weak)
        python main.py weak_supervision --model /home/tml/bo/CRAFT/CRAFT-Remade/model/synth/final_model.pkl --iterations 20
        ;;
    test_synth)
        python main.py test_synth --model /home/tml/bo/CRAFT/CRAFT-Remade/model/synth/final_model.pkl
        ;;
    bak)
        mkdir -p ../bak
        cp -r train_ctw ../bak
        cp -r src       ../bak
        cp main.py      ../bak
        cp config.py    ../bak
        cp .gitignore   ../bak
        cp -r .vscode   ../bak
        cp -r model     ../bak
        ;;
    *)
        echo "please input correct param."
esac
