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

#define DEVIATION 0.00001


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
     * FIXME refactor
     * @param chunk chunk
     */
    void insertSpots(float *chunk) {
        for (int i = 0; i < spot_size; ++i) {
            Spot spot = spots[i];
            if (spot.y >= myRank * block_height && spot.y < myRank * block_height + block_height) {
                chunk[getIndex(spot.x, spot.y - (myRank * block_height) + 1)] = spot.temperature;
            }
        }
    }

    bool computeInnerRows(const float *blockData, float *newBlock) {
        bool changeHappened = false;
        for (int y = 2; y < block_height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (newBlock[getIndex(x, y)] < 0) {

                    float surraundingSum = 0;

                    surraundingSum += blockData[getIndex(x, y)];
                    surraundingSum += blockData[getIndex(x, y - 1)];
                    surraundingSum += blockData[getIndex(x, y + 1)];

                    // check left wall
                    if (x > 0) {
                        surraundingSum += blockData[getIndex(x - 1, y)];
                        surraundingSum += blockData[getIndex(x - 1, y - 1)];
                        surraundingSum += blockData[getIndex(x - 1, y + 1)];
                    }
                    // check right wall
                    if (x < width - 1) {
                        surraundingSum += blockData[getIndex(x + 1, y)];
                        surraundingSum += blockData[getIndex(x + 1, y - 1)];
                        surraundingSum += blockData[getIndex(x + 1, y + 1)];
                    }

                    float value;
                    if (x > 0 && x < width - 1) {
                        value = surraundingSum / (float)9;
                    } else {
                        value = surraundingSum / (float)6;
                    }

                    newBlock[getIndex(x, y)] = value;

                    changeHappened |= fabs(value - blockData[getIndex(x, y)]) > DEVIATION;
                }
            }
        }
        return changeHappened;
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
                surraundingSum += returnIf(blockData[getIndex(x, y - 1)], myRank > 0, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x - 1, y)], x > 0, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y + 1)], x > 0, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y - 1)], x > 0 && myRank > 0, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x + 1, y)], x < width - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y + 1)], x < width - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y - 1)], x < width - 1 && myRank > 0,
                                           &numbersAdded);

                value = surraundingSum / numbersAdded;
                newBlock[getIndex(x, y)] = value;

                changeHappened |= fabs(value - blockData[getIndex(x, y)]) > DEVIATION;
            }

            // lower row
            y = block_height;
            surraundingSum = 0;
            numbersAdded = 0;

            if(newBlock[getIndex(x,y)] < 0) {
                surraundingSum += returnIf(blockData[getIndex(x, y)], true, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x, y + 1)], myRank < worldSize - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x, y - 1)], true, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x - 1, y)], x > 0, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y + 1)], x > 0 && myRank < worldSize - 1,
                                           &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x - 1, y - 1)], x > 0, &numbersAdded);

                surraundingSum += returnIf(blockData[getIndex(x + 1, y)], x < width - 1, &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y + 1)],
                                           x < width - 1 && myRank < worldSize - 1,
                                           &numbersAdded);
                surraundingSum += returnIf(blockData[getIndex(x + 1, y - 1)], x < width - 1, &numbersAdded);

                value = surraundingSum / numbersAdded;
                newBlock[getIndex(x, y)] = value;

                changeHappened |= fabs(value - blockData[getIndex(x, y)]) > DEVIATION;
            }
        }
        return changeHappened;
    }

    void startNodesComunication(float *blockData, MPI_Request &sendUp, MPI_Request &sendDown, MPI_Request &recUp,
                                MPI_Request &recDown) {
        if (myRank > 0) {
            // send first row to node above me
            MPI_Isend(blockData + width, width, MPI_FLOAT, myRank - 1, 1, MPI_COMM_WORLD, &sendUp);
            // receive last row from from node above me
            MPI_Irecv(blockData, width, MPI_FLOAT, myRank - 1, 1, MPI_COMM_WORLD, &recUp);
        }
        if (myRank < worldSize - 1) {
            // send last row to node below
            MPI_Isend(blockData + (width * block_height), width, MPI_FLOAT, myRank + 1, 1, MPI_COMM_WORLD,
                      &sendDown);
            // receive first row from from node below
            MPI_Irecv(blockData + (width * block_height) + width, width, MPI_FLOAT, myRank + 1, 1,
                      MPI_COMM_WORLD, &recDown);
        }
    }

    void waitForComnicationEnd(MPI_Request &recUp, MPI_Request &recDown) {
        if (myRank > 0) {
            MPI_Wait(&recUp, nullptr);
        }
        if (myRank < worldSize - 1) {
            MPI_Wait(&recDown, nullptr);
        }
    }

    float *computeOwnChunk() {

        // init chunks - add row above and under
        auto *chunk = (float *) calloc((block_height + 2) * width, sizeof(float));

        insertSpots(chunk);

        MPI_Request sendUp, recUp, sendDown, recDown;
        while (true) {
            // fixme refactor
            auto *new_chunk = (float *) malloc((block_height + 2) * width * sizeof(float));
            fillValue(new_chunk, (block_height + 2) * width, -1);

            startNodesComunication(chunk, sendUp, sendDown, recUp, recDown);

            insertSpots(new_chunk);

            int chunk_changes = computeInnerRows(chunk, new_chunk);

            waitForComnicationEnd(recUp, recDown);

            chunk_changes += computeOuterRows(chunk, new_chunk);

            // is it done yet?
            int all_changes = chunk_changes;
            recieveNumberOfChanges(&all_changes);
            if (all_changes < 1) {//if there is no difference, we are done for
                break;
            }

            chunk=new_chunk;
        }

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

/// main - Main method
    int compute(int argc, char **argv) {
        // Initialize MPI
        int initialised;
        MPI_Initialized(&initialised);
        if (!initialised)
            MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

        if (argc > 1) {
            // read the input instance
            auto mpiSpotType = createMpiSpotType();
            float *output; // linearized output

            if (myRank == 0) {
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

            if (myRank == 0) {
                string outputFileName(argv[2]);
                writeOutput(myRank, width, height, outputFileName, output);
            }
        } else {
            if (myRank == 0)
                cout << "Input instance is missing!!!\n" << endl;
        }
        MPI_Finalize();
        return 0;
    }

private:
    unsigned int width, height, block_height, spot_size;
    int worldSize, myRank;
    vector<Spot> spots;

};

int main(int argc, char **argv) {
    return HeatMap().compute(argc, argv);
}