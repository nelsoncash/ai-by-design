package cifar

import (
  "bufio"
  "encoding/binary"
  "fmt"
  "image"
  "os"
)

func ReadImageAsBytes(path string) (error, []byte) {
  file, err := os.Open(path)
  if err != nil {
    panic(err)
  }
  fileDetails, _ := file.Stat()
  fileLength := fileDetails.Size()
  bytes := make([]byte, fileLength)
  buf := bufio.NewReader(file)
  _, err = buf.Read(bytes)
  if err != nil {
    panic(err)
  }
  // fmt.Println(bytes)
  return nil, bytes
}

// from https://golang.org/pkg/image/#Image
func ConvertImageToRGBSlice(m image.Image) [][]uint8 {
  bounds := m.Bounds()

	// An image's bounds do not necessarily start at (0, 0), so the two loops start
	// at bounds.Min.Y and bounds.Min.X. Looping over Y first and X second is more
	// likely to result in better memory access patterns than X first and Y second.
  var total [][]uint8
  var rSlice, gSlice, bSlice []uint8
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := m.At(x, y).RGBA()
			// A color's RGBA method returns values in the range [0, 65535].
			// Shifting by 12 reduces this to the range [0, 255].
      // rgbSlice := []uint8{uint8(r>>8),uint8(g>>8),uint8(b>>8),uint8(a>>8)}
      // total = append(total, rgbSlice)
      rSlice = append(rSlice, uint8(r>>8))
      gSlice = append(gSlice, uint8(g>>8))
      bSlice = append(bSlice, uint8(b>>8))
		}
	}
  total = append(total, rSlice)
  total = append(total, gSlice)
  total = append(total, bSlice)

  // fmt.Println(total)
  return total
}

func ConvertToCifar(labelData uint8, imageData [][]uint8) []uint8 {
  var cifar []uint8
  cifar = append(cifar, labelData)
  for i := range imageData {
    cifar = append(cifar, imageData[i]...)
  }
  return cifar
}

func WriteCifar(cifar []uint8, filePath string) (string, error) {
  out, err := os.Create(filePath)
  if err != nil {
    return filePath, err
  }

  err = binary.Write(out, binary.LittleEndian, cifar)
  if err != nil {
    return filePath, err
  }

  return filePath, nil

}

func WriteImageAsCifar(m image.Image, filePath string, labelData uint8) error {
  imageData := ConvertImageToRGBSlice(m)
  cifar := ConvertToCifar(labelData, imageData)
  _, err := WriteCifar(cifar, filePath)
  if err != nil {
    fmt.Printf("Error writing to cifar out file: %s", err.Error())
  }
  return err
}
