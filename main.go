package main

import (
  "net/http"
  "strings"
  "fmt"
  "io/ioutil"
  //"github.com/disintegration/imaging"
  "image/color"
  "encoding/json"
  "image"
  "path/filepath"
  "os"
  "io"
  _ "image/jpeg"
  _ "image/gif"
  "image/png"
  "strconv"
  "time"
)

const (
  BASE_URL = "https://api.dribbble.com/v1"
  ITERATIONS = 20
)

var (
  TAGS = []string{
    "clean",
    "simple",
    "modern",
    "minimal",
    "vintage",
    "elegant",
    "royal",
    "retro",
    "flat",
    "nerd",
    "nerdy",
    "feminine",
    "woman",
    "male",
    "female",
    "man",
    "majestic",
    "power",
    "strong",
  }
  LOGO_SHOTS = []*Shot{}
  LOGO_SHOT_VECTORS = [][]int{}
  TOKEN = ""
  DB_ENTITIES = map[string]FilteredShot{}
)

// entities we will be writing to a file/db
type FilteredShot struct {
  Id        int `json:"id"`
  PhotoPath string `json:"photoPath"`
  TagVector []int  `json:"tagVector"`
}

// entity for storing any config variables
type Config struct {
  DribbbleKey string `json:"dribbbleKey"`
}

type Shot struct {
  Id            int       `json:"id"`
  Title         string    `json:"title"`
  Description   string    `json:"description"`
  Width         int       `json:"width"`
  Height        int       `json:"height"`
  ViewsCount    int       `json:"views_count"`
  ReboundsCount int       `json:"rebounds_count"`
  LikesCount    int       `json:"likes_count"`
  Images        Images    `json:"images"`
  Tags          []string  `json:"tags"`
}

type Images struct {
  Hidpi  string `json:"hidpi"`
  Normal string `json:"normal"`
  Teaser string `json:"teaser"`
}

// Converted implements image.Image, so you can
// pretend that it is the converted image.
type Converted struct {
    Img image.Image
    Mod color.Model
}

// We return the new color model...
func (c *Converted) ColorModel() color.Model{
    return c.Mod
}

// ... but the original bounds
func (c *Converted) Bounds() image.Rectangle{
    return c.Img.Bounds()
}

// At forwards the call to the original image and
// then asks the color model to convert it.
func (c *Converted) At(x, y int) color.Color{
    return c.Mod.Convert(c.Img.At(x,y))
}

func main() {
  initConfig()
  fetchPosts()
}

func initConfig() {
  var config Config
  configFile, err := os.Open("config.json")
  if err != nil {
    panic(err)
  }

  jsonParser := json.NewDecoder(configFile)
  if err = jsonParser.Decode(&config); err != nil {
    panic(err)
  }
  TOKEN = config.DribbbleKey
}

func (shot Shot) ProcessImage() {
  fmt.Println(shot.Images.Teaser)
  ext := filepath.Ext(shot.Images.Teaser)
  tmpFile, err := os.Create(strings.Join([]string{"tmp/", strconv.Itoa(shot.Id), ext}, ""))
  if err != nil {
    panic(err)
  }
  imgReq, err := http.NewRequest("GET", shot.Images.Teaser, nil)
  imgClient := &http.Client{}
  imgResp, err := imgClient.Do(imgReq)
  if err != nil {
    panic(err)
  }
  defer imgResp.Body.Close()
  _, err = io.Copy(tmpFile, imgResp.Body)
  if err != nil {
    panic(err)
  }
  tmpFile.Close()
  reader, err := os.Open(tmpFile.Name())
  if err != nil {
    panic(err)
  }
  defer reader.Close()
  m, _, err := image.Decode(reader)
  // fmt.Println(m)
  if m != nil {
    fmt.Println(m.Bounds())
  }
  if err != nil {
    panic(err)
  }

  convertToBW(strconv.Itoa(shot.Id), ext)
}

// this will convert an rgb image to grayscale
// this is more a utility function to test the algorithms efficacy at this point

func convertToBW(title, ext string) {
  path := strings.Join([]string{"tmp/", title, ext}, "")
  reader, err := os.Open(path)
  if err != nil {
    panic(err)
  }

  defer reader.Close()
  src, _, err := image.Decode(reader)
  // Since Converted implements image, this is now a grayscale image
  gr := &Converted{src, color.GrayModel}
  // Or do something like this to convert it into a black and
  // white image.
  // bw := []color.Color{color.Black,color.White}
  // gr := &Converted{src, color.Palette(bw)}
  outfile, err := os.Create(strings.Join([]string{"tmp-bw/", title, ext}, ""))
  if err != nil {
    panic(err)
  }
  defer outfile.Close()
  png.Encode(outfile,gr)
}

// Makes an HTTP request to Dribble's API
// grabs shots based on `week` parameters
func fetchFromDribble(queryString, path string) {
  headerValue := strings.Join([]string{"Bearer ", TOKEN}, "")
  url := strings.Join([]string{BASE_URL, path}, "/")
  url = strings.Join([]string{url, queryString}, "?")
  fmt.Println(url)
  req, err := http.NewRequest("GET", url, nil)
  req.Header.Set("Authorization", headerValue)
  client := &http.Client{}
  resp, err := client.Do(req)
  if err != nil {
    panic(err)
  }
  defer resp.Body.Close()
  var shots []*Shot
  body, _ := ioutil.ReadAll(resp.Body)

  err = json.Unmarshal(body, &shots)
  if err != nil {
    //return empty array of events if no results
    panic(err)
  }

  _ = filterShotsByTags(shots)
  fmt.Println(DB_ENTITIES)
  err = writeEntities()
  if err != nil {
    panic(err)
  }

  // only pull images from logo shots
  for _, shot := range LOGO_SHOTS {
    shot.ProcessImage()
  }
}

func filterShotsByTags(shots []*Shot) []*Shot {
  for _, shot := range shots {
    isLogo := false
    for i := range shot.Tags {
      if shot.Tags[i] == "logo" {
        isLogo = true
      }
    }
    if isLogo {
      LOGO_SHOTS = append(LOGO_SHOTS, shot)
      _, vector := containsAttribute(shot.Tags, TAGS)
      LOGO_SHOT_VECTORS = append(LOGO_SHOT_VECTORS, vector)
      filteredShot := FilteredShot{
        TagVector: vector,
        Id: shot.Id,
      }
      DB_ENTITIES[strconv.Itoa(shot.Id)] = filteredShot
    }
  }
  return shots
}

// Labels set up as 'one-hot vectors'
func containsAttribute(inputs []string, matches []string) (bool, []int) {
  matchAsVector := make([]int, len(matches))
  matched := false
  for a := range inputs {
    input := inputs[a]
    for i := range matches {
      if input == matches[i] {
        fmt.Println(input)
        matchAsVector[i] = 1
        matched = true
      }
    }
  }
  return matched, matchAsVector
}

func writeEntities() error {
  entitiesAsJson, err := json.Marshal(DB_ENTITIES)
  if err != nil {
    return err
  }
  ioutil.WriteFile("db/db.json", entitiesAsJson, 0644)
  return nil
}

func fetchPosts() {
  now := time.Now()
  for i := 0;i < ITERATIONS; i++ {
    then := now.AddDate(0, 0, (i * -7))
    y, m, d := then.Date()
    week := strings.Join([]string{strconv.Itoa(y), strconv.Itoa(int(m)), strconv.Itoa(d)}, "-")
    queryString := strings.Join([]string{"timeframe=", "weekly", "&date=", week}, "")
    fetchFromDribble(queryString, "shots")
  }
}
