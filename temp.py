
class MyNum:
    def __init__(self, val):
        self.val = val

    def __str__(self):
        ans = ""
        ans += f"H: {self.val:02x}H\n" # 0xFF, FFH
        ans += f"B: {self.val:08b}B\n" # 0b11111111, 11111111B
        ans += f"D: {self.val:d}D\n" # 255D
        return ans
    
    def left_shift(self, n):
        print(f"Left shift by {n} bits")
        self.val = self.val << n

    def right_shift(self, n):
        print(f"Right shift by {n} bits")
        self.val = self.val >> n

    def bitwise_or(self, b: 'MyNum'):
        print(f"Bitwise or with {b.val}")
        self.val = self.val | b.val

    def bitwise_and(self, b: 'MyNum'):
        print(f"Bitwise and with {b.val}")
        self.val = self.val & b.val

a = MyNum(16)
b = MyNum(32)
print(a)
print(b)

a.left_shift(8)
a.bitwise_or(b)
print(a)
