.PHONY: all clean
all:
	$(MAKE) -C lib/ctm-dist
	$(MAKE) -C lib/lda-c-dist
	$(MAKE) -C lib/hdp

clean:
	$(MAKE) -C lib/ctm-dist clean
	$(MAKE) -C lib/lda-c-dist clean
	$(MAKE) -C lib/hdp clean