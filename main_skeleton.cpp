#include <mpi.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <tuple>
#include <iostream>

using namespace std;

#define EPSILON 0.00001


/// @struct Spot
/// @brief The struct with parameters of a spot with given temperature
struct Spot {
    unsigned int x; ///< X-coordination of the spot
    unsigned int y; ///< y-coordination of the spot
    unsigned int temperature; ///< temperature of the spot

    /// operator== Comparison of spots from coordination point of view
    /// @param b - Spot for comparation
    bool operator==(const Spot &b) const {
        return (this->x == b.x) && (this->y == b.y);
    }
};

class HeatMap {
public:

/// readInstance - Method for reading the input instance file
/// @param instanceFileName - File name of the input instance
/// @return Tuple of (Width of the space; Height of the Space; Vector of the Spot)
    tuple<unsigned int, unsigned int, vector<Spot>> readInstance(const char *instanceFileName) {
        unsigned int width, height;
        vector<Spot> spots;
        string line;

        ifstream file(instanceFileName);
        if (file.is_open()) {
            int lineId = 0;
            while (std::getline(file, line)) {
                stringstream ss(line);
                if (lineId == 0) {
                    ss >> width;
                } else if (lineId == 1) {
                    ss >> height;
                } else {
                    unsigned int i, j, temperature;
                    ss >> i >> j >> temperature;
                    spots.push_back({i, j, temperature});
                }
                lineId++;
            }
            file.close();
        } else {
            throw runtime_error("It is not possible to open instance file!\n");
        }
        return make_tuple(width, height, spots);
    }

/// writeOutput - Method for creating resulting ppm image
/// @param myRank - Rank of the process
/// @param width - Width of the 2D space (image)
/// @param height - Height of the 2D space (image)
/// @param image - Linearized image
    void writeOutput(const int myRank, const int width, const int height, const string instanceFileName,
                     const float *image) {
        // Draw the output image
        ofstream file(instanceFileName);
        if (file.is_open()) {
            if (myRank == 0) {
                file << "P2\n" << width << "\n" << height << "\n" << 255 << "\n";
                for (unsigned long i = 0; i < width * height; i++) {
                    file << static_cast<int>(image[i]) << " ";
                }
            }
        }
        file.close();
    }

    void fillValue(float* data, int size, float value) {
        for (int i = 0; i < size; ++i) {
            data[i] = value;
        }
    }

    MPI_Datatype createMpiSpotType() {
        const int dataTypeSize = 3;
        int blocklengths[dataTypeSize] = {1, 1, 1};
        MPI_Datatype types[dataTypeSize] = {MPI_INT, MPI_INT, MPI_INT};
        MPI_Datatype mpiSpotType;
        MPI_Aint offsets[dataTypeSize];

        offsets[0] = offsetof(Spot, x);
        offsets[1] = offsetof(Spot, y);
        offsets[2] = offsetof(Spot, temperature);

        MPI_Type_create_struct(dataTypeSize, blocklengths, offsets, types, &mpiSpotType);
        MPI_Type_commit(&mpiSpotType);
        return mpiSpotType;
    }

    /**
     * Rank 0 sends init info to other nodes.
     */
    void sentInitInfo() {
        auto *info = new int[3];
        info[0] = width;
        info[1] = block_height;
        info[2] = spot_size;
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    }

    /**
     * Recieves init info from rank 0.
     */
    void recieveInitInfo() {
        auto *info = new int[3];
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        width = static_cast<unsigned int>(info[0]);
        block_height = static_cast<unsigned int>(info[1]);
        spot_size = static_cast<unsigned int>(info[2]);
    }

    /**
     * Sent spots to other nodes.
     * @param mpiSpotType spot MPI type
     */
    void sentSpots(MPI_Datatype mpiSpotType) {
        MPI_Bcast(spots.data(), spot_size, mpiSpotType, 0, MPI_COMM_WORLD);
    }

    /**
     * Sent spots to other nodes.
     * @param mpiSpotType spot MPI type
     */
    void recieveSpots(MPI_Datatype mpiSpotType) {
        auto recieved_spots = (Spot *) malloc(spot_size * sizeof(Spot));
        MPI_Bcast(recieved_spots, spot_size, mpiSpotType, 0, MPI_COMM_WORLD);

        for (int i = 0; i < spot_size; ++i) {
            spots.push_back(recieved_spots[i]);
        }
    }

    /**
     * Compiles all result chunks together to one output (which is on node with rank 0).
     * @param result_chunk result chunk
     * @param output overall output
     */
    void compileResults(float *result_chunk, float *output) {
        MPI_Gather(result_chunk + width, width * block_height, MPI_FLOAT, output, width * block_height, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    /**
     * Recieves number of changes from all rows.
     * @param all_changes all changes
     * @return number of chunk changes
     */
    int recieveNumberOfChanges(int *all_changes) {
        MPI_Allreduce(all_changes, all_changes, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    }

    int getIndex(int x, int y) { return y * width + x; }

    float returnIf(float value, bool condition, float *addedNumbers) {
        if (condition) {
            *addedNumbers += 1;
            return value;
        }
        return 0;
    }

    /**
     * Inserts spots into chunk.
     * @param chunk chunk
     */
    void insertSpots(float *chunk) {
        unsigned int chunk_start = own_rank * block_height;
        for (int i = 0; i < spot_size; ++i) {
            Spot spot = spots[i];
            if (spot.y >= chunk_start && spot.y < chunk_start + block_height) {
                chunk[getIndex(spot.x, spot.y - (chunk_start) + 1)] = spot.temperature;
            }
        }
    }

    /**
     * Computes inner rows of given chunk.
     * @param chunk chunk
     * @param new_chunk new chunk (calculated)
     * @return number of changes
     */
    int computeInnerRows(const float *chunk, float *new_chunk) {
        int number_of_changes = 0;
        for (int y = 2; y < block_height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (new_chunk[getIndex(x, y)] < 0) {

                    float sum = 0;
                    int n = 0;
                    int start_i = x > 0 ? - 1 : 0;
                    int end_i = x < width - 1 ? 1 : 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        for (int j = -1; j <= 1; ++j) {
                            sum += chunk[getIndex(x + i, y + j)];
                            n++;
                        }
                    }

                    float value = sum / (float) n;
                    new_chunk[getIndex(x, y)] = value;
                    number_of_changes += fabs(value - chunk[getIndex(x, y)]) > EPSILON ? 1 : 0;
                }
            }
        }

        return number_of_changes;
    }

    bool computeOuterRows(const float *blockData, float *newBlock) {
        bool changeHappened = false;
        for (int x = 0; x < width; ++x) {
            // upper row
            int y = 1;
            float surraundingSum = 0;
            float numbersAdded = 0;
            float value;

            if(newBlock[getIndex(x,y)] < 0) {
                surraundingSum += returnIf(blockData[getIndex(x, y)], true, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x, y + 1)], true, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x, y - 1)], own_rank > 0, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x - 1, y)], x > 0, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y + 1)], x > 0, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y - 1)], x > 0 && own_rank > 0, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x + 1, y)], x < width - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y + 1)], x < width - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y - 1)], x < width - 1 && own_rank > 0, &numbersAdded);

                value = surraundingSum / numbersAdded;
                newBlock[getIndex(x, y)] = value;

                changeHappened |= fabs(value - blockData[getIndex(x, y)]) > EPSILON;
            }

            // lower row
            y = block_height;
            surraundingSum = 0;
            numbersAdded = 0;

            if(newBlock[getIndex(x,y)] < 0) {
                surraundingSum += returnIf(blockData[getIndex(x, y)], true, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x, y + 1)], own_rank < worldSize - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x, y - 1)], true, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x - 1, y)], x > 0, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y + 1)], x > 0 && own_rank < worldSize - 1,
                                           &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y - 1)], x > 0, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x + 1, y)], x < width - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y + 1)], x < width - 1 && own_rank < worldSize - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y - 1)], x < width - 1, &numbersAdded);

                value = surraundingSum / numbersAdded;
                newBlock[getIndex(x, y)] = value;

                changeHappened |= fabs(value - blockData[getIndex(x, y)]) > EPSILON;
            }
        }
        return changeHappened;
    }

    /**
     * Sends first row to imeadiatelly preceding node and last node to immediately following node - if such exists.
     * Recieves rows in the same way.
     * @param chunk own chunk
     * @param send_upper_rows request for sending upper row
     * @param send_lower_rows request for sending lower row
     * @param recieve_upper_rows request for recieving upper row
     * @param recieve_lower_rows request for recieving lower row
     */
    void sendOuterRows(float *chunk, MPI_Request &send_upper_rows, MPI_Request &send_lower_rows, MPI_Request &recieve_upper_rows, MPI_Request &recieve_lower_rows) {
        if (own_rank > 0) {
            MPI_Isend(chunk + width, width, MPI_FLOAT, own_rank - 1, 1, MPI_COMM_WORLD, &send_upper_rows);
            MPI_Irecv(chunk, width, MPI_FLOAT, own_rank - 1, 1, MPI_COMM_WORLD, &recieve_upper_rows);
        }

        if (own_rank < worldSize - 1) {
            MPI_Isend(chunk + (width * block_height), width, MPI_FLOAT, own_rank + 1, 1, MPI_COMM_WORLD, &send_lower_rows);
            MPI_Irecv(chunk + (width * block_height) + width, width, MPI_FLOAT, own_rank + 1, 1, MPI_COMM_WORLD, &recieve_lower_rows);
        }
    }

    /**
     * Recieves edge rows from other nodes.
     * @param recieve_upper_row MPI request for upper row
     * @param recieve_lower_row MPI request for lower row
     */
    void recieveOuterRows(MPI_Request &recieve_upper_row, MPI_Request &recieve_lower_row) {
        if (own_rank > 0) {
            MPI_Wait(&recieve_upper_row, nullptr);
        }
        if (own_rank < worldSize - 1) {
            MPI_Wait(&recieve_lower_row, nullptr);
        }
    }

    /**
     * Computes heat diffusion in unknown number of iterations.
     * @return computed chunk
     */
    float *computeOwnChunk() {
        unsigned int chunk_length = (block_height + 2) * width;

        // init chunks - add row above and under
        auto *chunk = (float *) calloc(chunk_length, sizeof(float));
        insertSpots(chunk);
        auto *new_chunk = (float *) calloc(chunk_length, sizeof(float));

        MPI_Request sendUp, recieve_upper_row, sendDown, recieve_lower_row;

        int all_changes;
        do {
            // reset new chunk
            for (int i = 0; i < chunk_length; ++i) {
                new_chunk[i] = -1;
            }

            sendOuterRows(chunk, sendUp, sendDown, recieve_upper_row, recieve_lower_row);

            // insert spots to new chunk
            insertSpots(new_chunk);

            all_changes = computeInnerRows(chunk, new_chunk);

            recieveOuterRows(recieve_upper_row, recieve_lower_row);

            all_changes += computeOuterRows(chunk, new_chunk);

            // is it done yet?
            recieveNumberOfChanges(&all_changes);

            // switch pointers
            float *foo = chunk;
            chunk = new_chunk;
            new_chunk = foo;

        } while (all_changes >= 1);

        return chunk;
    }

    /**
     * Aligns problem height so all chunks are equal
     * @param original_size original problem size
     * @param number_of_nodes number of used nodes (cpus)
     * @return new size
     */
    unsigned int alignProblemSize(unsigned int original_size, int number_of_nodes) {
        if (original_size % number_of_nodes != 0) {
            return ((original_size / number_of_nodes) + 1) * number_of_nodes;
        } else {
            return original_size;
        }
    }


    /**
     * Main - assign work.
     * @param argc
     * @param argv
     * @return
     */
    int compute(int argc, char **argv) {
        // Initialize MPI
        int initialised;
        MPI_Initialized(&initialised);
        if (!initialised)
            MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);

        if (argc > 1) {
            // read the input instance
            auto mpiSpotType = createMpiSpotType();
            float *output; // linearized output

            if (own_rank == 0) {
                tie(width, height, spots) = readInstance(argv[1]);
                spot_size = static_cast<unsigned int>(spots.size());

                // allign the problem size
                height = alignProblemSize(height, worldSize);
                width = alignProblemSize(width, worldSize);

                // set working block height
                // computation is distributed by rows
                block_height = height / worldSize;

                sentInitInfo();

                // broadcast spots my_chunk_results
                sentSpots(mpiSpotType);

                // init output
                output = new float[width * height];

            } else {
                recieveInitInfo();

                recieveSpots(mpiSpotType);
            }

            auto *my_chunk_results = computeOwnChunk();

            compileResults(my_chunk_results, output);

            free(my_chunk_results);

            if (own_rank == 0) {
                string outputFileName(argv[2]);
                writeOutput(own_rank, width, height, outputFileName, output);
            }
        } else {
            if (own_rank == 0)
                cout << "Input instance is missing!!!\n" << endl;
        }
        MPI_Finalize();
        return 0;
    }

private:
    unsigned int width, height, block_height, spot_size;
    int worldSize, own_rank;
    vector<Spot> spots;

};

int main(int argc, char **argv) {
    return HeatMap().compute(argc, argv);
}