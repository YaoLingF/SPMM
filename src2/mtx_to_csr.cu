#include "mtx_to_csr.cuh"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

struct Entry
{
    int row;
    int col;
    float value;
};
} // namespace

CSR *load_mtx_to_csr(const char *filename)
{
    std::ifstream fin(filename);
    if (!fin)
    {
        throw std::runtime_error(std::string("failed to open MatrixMarket file: ") + filename);
    }

    std::string line;
    if (!std::getline(fin, line))
    {
        throw std::runtime_error("empty MatrixMarket file");
    }

    std::stringstream banner(line);
    std::string banner_head, object, format, field, symmetry;
    banner >> banner_head >> object >> format >> field >> symmetry;
    banner_head = lower_copy(banner_head);
    object = lower_copy(object);
    format = lower_copy(format);
    field = lower_copy(field);
    symmetry = lower_copy(symmetry);

    if (banner_head != "%%matrixmarket" || object != "matrix" || format != "coordinate")
    {
        throw std::runtime_error("only MatrixMarket coordinate matrices are supported");
    }
    if (field == "complex")
    {
        throw std::runtime_error("complex MatrixMarket matrices are not supported");
    }

    bool pattern = (field == "pattern");
    bool symmetric = (symmetry == "symmetric" || symmetry == "hermitian");
    if (symmetry == "skew-symmetric")
    {
        throw std::runtime_error("skew-symmetric MatrixMarket matrices are not supported");
    }

    do
    {
        if (!std::getline(fin, line))
            throw std::runtime_error("missing MatrixMarket size line");
    } while (line.empty() || line[0] == '%');

    int rows = 0;
    int cols = 0;
    int meta_nnz = 0;
    {
        std::stringstream sizes(line);
        sizes >> rows >> cols >> meta_nnz;
    }
    if (rows <= 0 || cols <= 0 || meta_nnz < 0)
    {
        throw std::runtime_error("invalid MatrixMarket size line");
    }

    std::vector<Entry> entries;
    entries.reserve(symmetric ? meta_nnz * 2 : meta_nnz);

    int seen = 0;
    while (std::getline(fin, line))
    {
        if (line.empty() || line[0] == '%')
            continue;

        std::stringstream ss(line);
        int r = 0;
        int c = 0;
        float v = 1.0f;
        if (pattern)
        {
            ss >> r >> c;
        }
        else
        {
            ss >> r >> c >> v;
        }
        if (!ss)
        {
            throw std::runtime_error("invalid MatrixMarket entry line");
        }

        r -= 1;
        c -= 1;
        if (r < 0 || r >= rows || c < 0 || c >= cols)
        {
            throw std::runtime_error("MatrixMarket entry index out of range");
        }

        entries.push_back({r, c, v});
        if (symmetric && r != c)
        {
            entries.push_back({c, r, v});
        }
        seen++;
    }
    if (seen != meta_nnz)
    {
        throw std::runtime_error("MatrixMarket nnz count does not match header");
    }

    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
    });

    std::vector<Entry> merged;
    merged.reserve(entries.size());
    for (const Entry &entry : entries)
    {
        if (!merged.empty() && merged.back().row == entry.row && merged.back().col == entry.col)
        {
            merged.back().value += entry.value;
        }
        else
        {
            merged.push_back(entry);
        }
    }

    CSR *csr = static_cast<CSR *>(malloc(sizeof(CSR)));
    csr->nRow = rows;
    csr->nCol = cols;
    csr->nnz = static_cast<int>(merged.size());
    csr->row_offset = static_cast<int *>(malloc(sizeof(int) * (rows + 1)));
    csr->col_idx = static_cast<int *>(malloc(sizeof(int) * csr->nnz));
    csr->value = static_cast<half *>(malloc(sizeof(half) * csr->nnz));

    std::fill(csr->row_offset, csr->row_offset + rows + 1, 0);
    for (const Entry &entry : merged)
    {
        csr->row_offset[entry.row + 1]++;
    }
    for (int r = 0; r < rows; r++)
    {
        csr->row_offset[r + 1] += csr->row_offset[r];
    }

    std::vector<int> cursor(csr->row_offset, csr->row_offset + rows);
    for (const Entry &entry : merged)
    {
        int dst = cursor[entry.row]++;
        csr->col_idx[dst] = entry.col;
        csr->value[dst] = __float2half(entry.value);
    }

    return csr;
}
