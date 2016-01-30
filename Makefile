.PHONY: all

all:
	git add --all
	git commit -a -m "make all"
	git pull
	git push
