//! @file Simple 2D Heat Diffusion simulator

#include <mpi.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <tuple>

using namespace std;

/// @struct Spot
/// @brief The struct with parameters of a spot with given temperature
struct Spot {
    unsigned int x; ///< X-coordination of the spot
    unsigned int y; ///< y-coordination of the spot
    unsigned int temperature; ///< temperature of the spot

    /// operator== Comparison of spots from coordination point of view
    /// @param b - Spot for comparation
    bool operator==(const Spot& b) const
    {
        return (this->x == b.x) && (this->y == b.y);
    }
};

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
            }
            else if (lineId == 1) {
                ss >> height;
            }
            else {
                unsigned int i, j, temperature;
                ss >> i >> j >> temperature;
                spots.push_back({i, j, temperature});
            }
            lineId++;
        }
        file.close();
    }
    else {
        throw runtime_error("It is not possible to open instance file!\n");
    }
    return make_tuple(width, height, spots);
}

/// writeOutput - Method for creating resulting ppm image
/// @param myRank - Rank of the process
/// @param width - Width of the 2D space (image)
/// @param height - Height of the 2D space (image)
/// @param image - Linearized image
void writeOutput(const int myRank, const int width, const int height, const string instanceFileName, const vector<int> image){
  // Draw the output image
  ofstream file(instanceFileName);
  if (file.is_open())
  {
    if (myRank == 0) {
      file << "P2\n" << width << "\n" << height << "\n" << 255 <<  "\n";
      for (unsigned long i = 0; i < image.size(); i++) {
        file << image[i] << " ";
      }
    }
  }
  file.close();
}

/// main - Main method
int main(int argc, char **argv) {
    // Initialize MPI
    int worldSize, myRank;
    int initialised;
    MPI_Initialized(&initialised);
    if(!initialised)
        MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (argc > 1) {
        // read the input instance
        unsigned int width, height, spotsSize;
        vector<Spot> spots;
        vector<int> image; // linearized image
        if (myRank == 0) {
            tie(width, height, spots) = readInstance(argv[1]);
            spotsSize = static_cast<unsigned int>(spots.size());

            // round the spacesize of the image to be multiple of the number of used processes
            if (height % worldSize != 0) height = ((height / worldSize) + 1) * worldSize;
            if (width % worldSize != 0) width = ((width/worldSize) + 1) * worldSize;
        }

        //-----------------------\\
        // Insert your code here \\
        //        |  |  |        \\
        //        V  V  V        \\



        //-----------------------\\

       	if(myRank == 0)
       	{
        	string outputFileName(argv[2]);
        	writeOutput(myRank, width, height, outputFileName, image);
    	}
    }
    else {
        if (myRank == 0)
            cout << "Input instance is missing!!!\n" << endl;
    }
    MPI_Finalize();
    return 0;
}
