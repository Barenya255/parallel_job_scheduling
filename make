# Makefile

# Compiler 
NVCC = nvcc

# cuda file name
TARGET = parallel_fcfs_job_sched

# Specify the source file
SOURCE = parallel_fcfs_job_sched.cu

# Specify the input directory containing .txt files
INPUT_DIR = ./input

# Build rule
all: $(TARGET)

$(TARGET): $(SOURCE)
    $(NVCC) $(SOURCE) -o $(TARGET)

# Run rule
run: $(TARGET)
    @for file in $(INPUT_DIR)/*.txt; do \
        echo "Running with input file $$file"; \
        ./$(TARGET) $$file; \
    done

# Clean rule
clean:
    rm -f $(TARGET)

.PHONY: all run clean
