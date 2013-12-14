
all: wolfe-finalproject.pdf

wolfe-finalproject.pdf: clean
	pdflatex naaclhlt2013
	bibtex naaclhlt2013
	pdflatex naaclhlt2013
	pdflatex naaclhlt2013
	mv naaclhlt2013.pdf wolfe-finalproject.pdf

clean:
	rm -f *.blg
	rm -f *.bbl
	rm -f *.aux
	rm -f *.log
	rm -f *.pdf

