.PHONY: clean docs

clean:
	rm -rf docs/build/

docs:
	sphinx-build -b html docs/source/ docs/build/html