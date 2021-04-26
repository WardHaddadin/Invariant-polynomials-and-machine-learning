//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Outer products ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline std::vector<std::vector<T>> operator^(const std::vector<T>& a, const std::vector<T>& b) {
    
    std::vector<std::vector<T>> c(a.size(), std::vector<T>(b.size()));
    
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < b.size(); j++)
        {
            c[i][j] = a[i] * b[j];
        }
    }
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Matrix * Matrix ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {

    if (a[0].size() != b.size())
            throw std::invalid_argument("Matrix/Matix wrong size for matrix product, " + std::to_string(a.size()) + " x " + std::to_string(a[0].size()) + " and " +  std::to_string(b.size()) + " x " + std::to_string(b[0].size()) );

    std::vector<std::vector<T>> c(a.size(), std::vector<T>(b[0].size()));

    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < b[0].size(); j++)
        {
            c[i][j] = 0;
            for(int k = 0; k < b.size(); k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Matrix * vector ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator*(const std::vector<std::vector<T>>& a, const std::vector<T>& b) {
    
    if (a[0].size() != b.size())
            throw std::invalid_argument("Matrix/Vector wrong size for matrix * vec, " + std::to_string(a.size()) + " x " + std::to_string(a[0].size()) + " and " + std::to_string(b.size()));
    
    std::vector<T> c(a.size());
    
    for(int i = 0; i < a.size(); i++)
    {
        c[i] = 0;
        for(int j = 0; j < a[i].size(); j++)
        {
            c[i] += a[i][j] * b[j];
        }
    }
    
    return c;
}

template <typename T>
inline std::vector<T> operator*(const std::vector<T>& a, const std::vector<std::vector<T>>& b) {
    
    if (a.size() != b.size())
            throw std::invalid_argument("Matrix/Vector wrong size for vec^T * Matrix, " + std::to_string(b.size()) + " x " + std::to_string(b[0].size()) + " and " + std::to_string(a.size()));
    
    std::vector<T> c(b[0].size());
    
    for(int i = 0; i < b[0].size(); i++)
    {
        c[i] = 0;
        for(int j = 0; j < a.size(); j++)
        {
            c[i] += a[j] * b[j][i];
        }
    }
    
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// scalar * vector ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator*(const double& a, const std::vector<T>& b) {
    
    std::vector<T> c(b.size());
    
    for(int i = 0; i < b.size(); i++)
        c[i] = a * b[i];
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////vector / Matrix * vector / Matrix element wise////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator%( const std::vector<T>& a, const std::vector<T>& b) {
    
    if (a.size() != b.size())
            throw std::invalid_argument("Vector/Vector wrong size for element wise prod, " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
    
    std::vector<T> c(a.size());
    
    for(int i = 0; i < b.size(); i++)
        c[i] = a[i] * b[i];
    
    return c;
}

template <typename T>
inline std::vector<std::vector<T>> operator%( const std::vector<std::vector<T>>& a, const std::vector<T>& b) {
    
    if (a[0].size() != b.size())
            throw std::invalid_argument("Vector/Vector wrong size for element wise prod, " + std::to_string(a.size()) + " x " + std::to_string(a[0].size())  + " and " + std::to_string(b.size()));
    
    std::vector<std::vector<T>> c(a.size(), std::vector<T>(b.size()));
    
    for(int m = 0; m < a.size(); m++)
        for(int i = 0; i < b.size(); i++)
            c[m][i] = a[m][i] * b[i];
    
    return c;
}

template <typename T>
inline std::vector<std::vector<T>> operator%(const std::vector<T>& a, const std::vector<std::vector<T>>& b) {
    
    if (a.size() != b.size())
            throw std::invalid_argument("Vector/Matrix wrong size for element wise prod, " + std::to_string(a.size())  + " and " + std::to_string(b.size()) + " x " + std::to_string(b[0].size()) ) ;
    
    std::vector<std::vector<T>> c(a.size(), std::vector<T>(b[0].size()));
    
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < b[i].size(); j++)
        {
            c[i][j] = a[i] * b[i][j];
        }
    }
    
    return c;
}

template <typename T>
inline std::vector<std::vector<std::vector<T>>> operator%(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
    
    if (a[0].size() != b.size())
            throw std::invalid_argument("Matrix/Matrix wrong size for element wise prod, " + std::to_string(a.size()) + " x " + std::to_string(a[0].size())  + " and " + std::to_string(b.size()) + " x " + std::to_string(b[0].size()) ) ;
    
    std::vector<std::vector<std::vector<T>>> c(a.size(), std::vector<std::vector<T>>(a[0].size(), std::vector<T>(b[0].size())));
    
    for(int m = 0; m < a.size(); m++)
    {
        for(int i = 0; i < a[0].size(); i++)
        {
            for(int j = 0; j < b[i].size(); j++)
            {
                c[m][i][j] = a[m][i] * b[i][j];
            }
        }
    }
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////dot product ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T operator*( const std::vector<T>& a, const std::vector<T>& b) {
    
    if (a.size() != b.size())
            throw std::invalid_argument("Vectors wrong size for dot product, " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
    
    T c = 0;
    
    for(int i = 0; i < b.size(); i++)
        c += a[i] * b[i];
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////vector / scalar ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator/(const std::vector<T>& b, const double& a) {
    
    std::vector<T> c(b.size());
    
    for(int i = 0; i < b.size(); i++)
        c[i] = b[i] / a;
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////vector sum ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b) {
    
    if (a.size() != b.size())
            throw std::invalid_argument("Vectors wrong size for sum, " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
    
    std::vector<T> c(a.size());
    
    for(int i = 0; i < a.size(); i++)
        c[i] = a[i] + b[i];
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////vector subtract ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) {
    
    if (a.size() != b.size())
            throw std::invalid_argument("Vectors wrong size for subtraction, " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
    
    std::vector<T> c(a.size());
    
    for(int i = 0; i < a.size(); i++)
        c[i] = a[i] - b[i];
    
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////negative vector ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> operator-(const std::vector<T>& a) {
    
    std::vector<T> c(a.size());
    
    for(int i = 0; i < a.size(); i++)
        c[i] = -a[i];
    
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// output vector ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& D)
{
    for(int i = 0; i < D.size(); i++)
    {
        os << D[i] << " ";
    }
  return os;
}
