include .env
export

.PHONY: gen-reader

# Generate ccl_science_data/gen_reader_ext.py from Rust source files in rankless_rs.
gen-reader:
	python scripts/gen_reader.py --gen-root $(RANKLESS_APP_ROOT)
