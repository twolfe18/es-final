
all: clean wolfe-finalproject.pdf

wolfe-finalproject.pdf: learningCurve.pdf
	pdflatex naaclhlt2013
	bibtex naaclhlt2013
	pdflatex naaclhlt2013
	pdflatex naaclhlt2013
	mv naaclhlt2013.pdf wolfe-finalproject.pdf

learningCurve.pdf: ../vanilla.log ../fadditive.log
	 grep "loss on" ../vanilla.log | awk '{print $$7}' > vanilla.learning
	 grep "loss on" ../fadditive.log | awk '{print $$7}' > fadditive.learning
	 paste vanilla.learning fadditive.learning | awk '{if(NF==1)$$2="NA"; print $$0}' >learning.txt
	 rm vanilla.learning fadditive.learning
	 Rscript figure.r

clean:
	rm -f *.blg
	rm -f *.bbl
	rm -f *.aux
	rm -f *.log
	rm -f *.out
	rm -f *.pdf
	rm -f learningCurve.pdf

