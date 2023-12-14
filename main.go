package main

import (
	"image"
	"image/color"
	"image/gif"
	"log"
	"os"

	"github.com/mazznoer/colorgrad"
	"gonum.org/v1/gonum/mat"
)

const (
	n      = 512
	frames = 1024
	delay  = 1
	delayS = float64(delay) / 100.0

	diff = 0.001
	visc = 0.0001
)

func swap(x, y *mat.Dense) {
	tmp := mat.DenseCopyOf(x)
	x.Copy(y)
	y.Copy(tmp)
}

func setBound(b int, x *mat.Dense) {
	for i := 1; i <= n; i++ {
		if b == 1 {
			x.Set(0, i, -x.At(1, i))
			x.Set(n+1, i, -x.At(n, i))
		} else {
			x.Set(0, i, x.At(1, i))
			x.Set(n+1, i, x.At(n, i))
		}
		if b == 2 {
			x.Set(i, 0, -x.At(i, 1))
			x.Set(i, n+1, -x.At(i, n))
		} else {
			x.Set(i, 0, x.At(i, 1))
			x.Set(i, n+1, x.At(i, n))
		}
	}
}

func addSource(x, s *mat.Dense, dt float64) {
	tmp := mat.DenseCopyOf(s)
	tmp.Scale(dt, tmp)
	x.Add(x, tmp)
}

func diffuse(b int, x, x0 *mat.Dense, diffusion, dt float64) {
	a := dt * diffusion * n * n
	for k := 0; k < 20; k++ {
		for i := 1; i <= n; i++ {
			for j := 1; j <= n; j++ {
				x.Set(i, j, (x0.At(i, j)+a*(x.At(i-1, j)+x.At(i+1, j)+x.At(i, j-1)+x.At(i, j+1)))/(1+4*a))
			}
		}
		setBound(b, x)
	}
}

func advect(b int, d, d0, u, v *mat.Dense, dt float64) {
	dt0 := dt * n
	for i := 1; i <= n; i++ {
		for j := 1; j <= n; j++ {
			x := float64(i) - dt0*u.At(i, j)
			y := float64(j) - dt0*v.At(i, j)
			if x < 0.5 {
				x = 0.5
			}
			if x > n+0.5 {
				x = n + 0.5
			}
			i0 := int(x)
			i1 := i0 + 1
			if y < 0.5 {
				y = 0.5
			}
			if y > n+0.5 {
				y = n + 0.5
			}
			j0 := int(y)
			j1 := j0 + 1
			s1 := x - float64(i0)
			s0 := 1 - s1
			t1 := y - float64(j0)
			t0 := 1 - t1
			d.Set(i, j, s0*(t0*d0.At(i0, j0)+t1*d0.At(i0, j1))+s1*(t0*d0.At(i1, j0)+t1*d0.At(i1, j1)))
		}
	}
	setBound(b, d)
}

func project(u, v, p, div *mat.Dense) {
	h := 1.0 / n
	for i := 1; i <= n; i++ {
		for j := 1; j <= n; j++ {
			div.Set(i, j, -0.5*h*(u.At(i+1, j)-u.At(i-1, j)+v.At(i, j+1)-v.At(i, j+1)))
			p.Set(i, j, 0)
		}
	}
	setBound(0, div)
	setBound(0, p)
	for k := 0; k < 20; k++ {
		for i := 1; i <= n; i++ {
			for j := 1; j <= n; j++ {
				p.Set(i, j, (div.At(i, j)+p.At(i-1, j)+p.At(i+1, j)+p.At(i, j-1)+p.At(i, j+1))/4)
			}
		}
		setBound(0, p)
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= n; j++ {
			u.Set(i, j, u.At(i, j)-0.5*(p.At(i+1, j)-p.At(i-1, j))/h)
			v.Set(i, j, v.At(i, j)-0.5*(p.At(i, j+1)-p.At(i, j-1))/h)
		}
	}
	setBound(1, u)
	setBound(2, v)
}

func densityStep(x, x0, u, v *mat.Dense, dt float64) {
	addSource(x, x0, dt)
	swap(x0, x)
	diffuse(0, x, x0, diff, dt)
	swap(x0, x)
	advect(0, x, x0, u, v, dt)
}

func velocityStep(u, v, u0, v0 *mat.Dense, dt float64) {
	addSource(u, u0, dt)
	addSource(v, v0, dt)
	swap(u0, u)
	swap(v0, v)
	diffuse(1, u, u0, visc, dt)
	diffuse(2, v, v0, visc, dt)
	project(u, v, u0, v0)
	swap(u0, u)
	swap(v0, v)
	advect(1, u, u0, u0, v0, dt)
	advect(2, v, v0, u0, v0, dt)
	project(u, v, u0, v0)
}

func Simulate(d, d0, u, u0, v, v0 *mat.Dense, op func(i int, d, u, v *mat.Dense), c chan<- *mat.Dense) {
	for i := 0; i < frames; i++ {
		d0.Copy(d)
		u0.Copy(u)
		v0.Copy(v)
		velocityStep(u, v, u0, v0, delayS)
		densityStep(d, d0, u, v, delayS)
		op(i, d, u, v)
		c <- mat.DenseCopyOf(d)
	}
	close(c)
}

func main() {
	anim := gif.GIF{LoopCount: frames}
	pal := []color.Color{}
	grad := colorgrad.Viridis()
	for _, c := range grad.Colors(256) {
		pal = append(pal, c)
	}

	var (
		density     = mat.NewDense(n+2, n+2, nil)
		densityPrev = mat.NewDense(n+2, n+2, nil)
		u           = mat.NewDense(n+2, n+2, nil)
		uPrev       = mat.NewDense(n+2, n+2, nil)
		v           = mat.NewDense(n+2, n+2, nil)
		vPrev       = mat.NewDense(n+2, n+2, nil)
	)

	density.Apply(func(i, j int, _ float64) float64 { return 0.0 }, density)
	u.Apply(func(i, j int, _ float64) float64 { return 0.5 }, density)
	v.Apply(func(i, j int, _ float64) float64 { return 0.5 }, density)

	c := make(chan *mat.Dense, 16)
	go Simulate(density, densityPrev, u, uPrev, v, vPrev, func(i int, d, u, v *mat.Dense) {
		for x := 0; x < 64; x++ {
			for y := 0; y < 64; y++ {
				d.Set(16+x, 16+y, 1.0)
				d.Set(n-15-x, n-15-y, 0.0)
			}
		}
		for x := 0; x < n+2; x++ {
			for y := 0; y < n+2; y++ {
				if d.At(x, y) > 1 {
					d.Set(x, y, 1)
				} else if d.At(x, y) < 0 {
					d.Set(x, y, 0)
				}
			}
		}
	}, c)

	i := 0
	for {
		d, ok := <-c
		if !ok {
			break
		}
		img := image.NewPaletted(image.Rect(0, 0, n, n), pal)
		for x := 0; x < n; x++ {
			for y := 0; y < n; y++ {
				img.SetColorIndex(x, y, uint8(d.At(x+1, y+1)*255))
			}
		}

		anim.Delay = append(anim.Delay, delay)
		anim.Image = append(anim.Image, img)

		if ((i + 1) & i) == 0 {
			log.Printf("Frame %v", i+1)
		}
		i++
	}

	log.Printf("Completed %v frames.", len(anim.Image))

	gif.EncodeAll(os.Stdout, &anim)
}
