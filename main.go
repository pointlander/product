// Copyright 2025 The Product Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"strconv"

	"github.com/pointlander/product/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

var (
	// FlagCluster cluster mode
	FlagCluster = flag.Bool("cluster", false, "cluster mode")
)

// ClusterMode cluster mode
func ClusterMode() {
	data := Load()
	products := make([][]float64, len(data))
	for i := range products {
		column := NewMatrix(4, 1, make([]float64, 4)...)
		for ii := range data[i].Measures {
			column.Data[ii] = data[i].Measures[ii]
		}
		tensor := column.Tensor(column)
		ranks := PageRank(1.0, 1024, 1, tensor)
		products[i] = ranks.Data
	}
	for i := range products {
		fmt.Println(products[i])
	}
	vectors := make([][]float64, len(data))
	for i := range vectors {
		vectors[i] = make([]float64, 4)
	}
	results := make([][]float64, 4)
	for i := range 4 {
		column := NewMatrix(len(data), 1, make([]float64, len(data))...)
		for ii := range data {
			column.Data[ii] = products[ii][i]
		}
		tensor := column.Tensor(column)
		ranks := PageRank(1.0, 16, 1, tensor)
		results[i] = ranks.Data
		fmt.Println(ranks)
		for ii, value := range ranks.Data {
			vectors[ii][i] = value
		}
	}
	meta := make([][]float64, len(vectors))
	for i := range meta {
		meta[i] = make([]float64, len(vectors))
	}
	const k = 3
	for i := 0; i < 33; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range data {
		fmt.Println(clusters[i], data[i].Label)
	}
	a := make(map[string][3]int)
	for i := range vectors {
		histogram := a[data[i].Label]
		histogram[clusters[i]]++
		a[data[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}

	{
		average := make([]float64, len(vectors))
		for _, result := range results {
			for ii, value := range result {
				average[ii] += value
			}
		}
		for i := range average {
			average[i] /= float64(len(results))
		}

		cov := make([][]float64, len(vectors))
		for i := range cov {
			cov[i] = make([]float64, len(vectors))
		}
		for _, measures := range results {
			for i, v := range measures {
				for ii, vv := range measures {
					diff1 := average[i] - v
					diff2 := average[ii] - vv
					cov[i][ii] += diff1 * diff2
				}
			}
		}
		if len(results) > 0 {
			for i := range cov {
				for ii := range cov[i] {
					cov[i][ii] = cov[i][ii] / float64(len(results))
				}
			}
		}

		meta := make([][]float64, len(vectors))
		for i := range meta {
			meta[i] = make([]float64, len(vectors))
		}
		const k = 3
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), cov, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
		clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := range data {
			fmt.Println(clusters[i], data[i].Label)
		}
		a := make(map[string][3]int)
		for i := range vectors {
			histogram := a[data[i].Label]
			histogram[clusters[i]]++
			a[data[i].Label] = histogram
		}
		for k, v := range a {
			fmt.Println(k, v)
		}
	}
}

func main() {
	flag.Parse()

	if *FlagCluster {
		ClusterMode()
		return
	}
}
