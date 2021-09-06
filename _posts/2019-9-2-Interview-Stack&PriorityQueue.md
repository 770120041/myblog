---
layout: post
title:  Stack&PriorityQueue
categories: Interview
---

## 155. Min Stack
#### brute force but not constant time
```
class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {
        
    }
    
    void push(int x) {
        minV = min(minV,x);
        nums.push_back(x);
    }
    
    void pop() {
        if(nums.back() == minV){
            minV = INT_MAX;
            for(int i=0;i<nums.size()-1;i++) minV = min(minV,nums[i]);
        }
        nums.pop_back();
    }
    
    int top() {
        return nums.back();
    }
    
    int getMin() {
        return minV;
    }
private:
    vector<int> nums;
    int minV = INT_MAX;
};
```
#### Tricky - 2 stack
Using two stacks, the reason why this works is because larger values appears later will never be used as the min value.
```
class MinStack {
public:
    MinStack() {
    }
    
    void push(int x) {
        s1.push(x);
        if(s2.empty() or s2.top() >= x){
            s2.push(x);
        }
    }
    
    void pop() {
        if(s1.top() == s2.top()){
            s2.pop();
        }
        s1.pop();
    }
    
    int top() {
        return s1.top();
    }
    
    int getMin() {
        return s2.top();
    }
private:
    stack<int> s1,s2;
};

 ```

 #### One stack
 We can store the min value temporaryly inside the stack
 ```
 class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {
        minVal = INT_MAX;
    }
    
    void push(int x) {
        if(x <= minVal){
            s.push(minVal);
            minVal = x;
        }
        s.push(x);
    }
    
    void pop() {
        int t = s.top(); s.pop();
        if(t == minVal){
            minVal = s.top(); s.pop();
        }
    }
    
    int top() {
        return s.top();
    }
    
    int getMin() {
        return minVal;
    }
    stack<int> s;
    int minVal;
};
```

 <hr>


## 225. Implement Stack using Queues
#### Brute force
each pop cost O(N) time complexity
```
class MyStack {
public:
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        if(q1.empty()){
        }
        else{
            while(!q1.empty()){
                q2.push(q1.front()); q1.pop();
            }
        }
        q1.push(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int ret = q1.front(); q1.pop();
        while(!q2.empty() and q2.size()>1){
            q1.push(q2.front());
            q2.pop();
        }
        swap(q1,q2);
        return ret;
    }
    
    /** Get the top element. */
    int top() {
        return q1.front();
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return q1.empty() and q2.empty();
    }
private:
    queue<int> q1,q2;
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
 ```

#### Another solution
Make the value appeared later on the top of stack, push(O(N))
```
class MyStack {
public:
    /** Initialize your data structure here. */
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        q2.push(x);
        while(!q1.empty()){
            q2.push(q1.front()); q1.pop();
        }
        swap(q1,q2);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int ret = q1.front();
        q1.pop();
        return ret;
    }
    
    /** Get the top element. */
    int top() {
        return q1.front();
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return q1.empty();
    }
    queue<int> q1,q2;
};
```

<hr>


## 232. Implement Queue using Stacks
#### Brute force
Put new elements in the bottom of stack
```
class MyQueue {
public:
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        while(!s1.empty()){
            s2.push(s1.top()); s1.pop();
        }
        s2.push(x);
        while(!s2.empty()){
            s1.push(s2.top());
            s2.pop();
        }
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        int ret = s1.top();
        s1.pop();
        return ret;
    }
    
    /** Get the front element. */
    int peek() {
        return s1.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return s1.empty();
    }
    stack<int> s1,s2;
};
```

<hr>

## 150. Evaluate Reverse Polish Notation
string to int: `stoi`, string to float `stof`
```class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> numbers;
        for(int i=0;i<tokens.size();i++){
            if(tokens[i] == "+" or tokens[i] == "-" or tokens[i] == "*" or tokens[i] == "/"){
                int num2 = numbers.top(); numbers.pop(); int num1 = numbers.top(); numbers.pop();
                char c = tokens[i][0];
                switch(c){
                    case '+':
                        numbers.push(num1+num2);    
                        break;
                    case '-':
                        numbers.push(num1-num2);    
                        break;
                    case '*':
                        numbers.push(num1*num2);    
                        break;
                    case '/':
                        numbers.push(num1/num2);    
                        break;
                }
            }
            else{
                numbers.push(stoi(tokens[i]));
            }
        }
        return numbers.top();
    }
};
```

<hr>

## 中缀,前缀，后缀表达式
[blog](https://blog.csdn.net/sgbfblog/article/details/8001651)
[blog](https://blog.csdn.net/walkerkalr/article/details/22798365)
<hr>

## 224 Basic Calculator && 227 Basic Calculator II && 	772 Basic Calculator III 
```cpp
class Solution {
public:
    void calWithOp(stack<int>&nums,char op){
        int result;
        int num2 = nums.top(); nums.pop();
        int num1 = nums.top(); nums.pop();
        switch(op){
            case '+': result = num1+num2; break;
            case '-': result = num1-num2; break;
            case '*': result = num1*num2; break;
            case '/': result = num1/num2; break;
        }
        // printf("num1=%d,num2=%d,op=%c,result=%d\n",num1,num2,op,result);
        nums.push(result);
    }
    void parseOps(stack<int> &nums,stack<char> &ops,char curOp){
        unordered_map<char,int> uMap{{'(',0},{'+',1},{'-',1},{'*',2},{'/',2}};
        if(curOp == ')'){
            while(ops.top() != '('){
                char prevOP = ops.top();
                ops.pop();
                calWithOp(nums,prevOP);
            }
            ops.pop();
        }
        else{
            while(!ops.empty()){
                char prevOP = ops.top();
                if(uMap[prevOP] and uMap[curOp] and uMap[curOp] <= uMap[prevOP]){
                    ops.pop();
                    calWithOp(nums,prevOP);
                }
                else{
                    break;
                }
            }
            ops.push(curOp);
        }
    }
    
    int calculate(string s) {
        string tmp;
        for(int i=0;i<s.size();i++) if(s[i] != ' ') tmp.push_back(s[i]);
        s = tmp;
        if(s.empty()) return 0;
        stack<int> nums;
        stack<char> ops;
        int i = 0;
        while(i < s.size()){
            // negative is 0 - it
            if(s[i] >= '0' and s[i] <= '9'){
                long long x = 0;
                while(i<s.size() and (s[i]>='0' and s[i]<='9')){
                    x = x*10 + s[i]-'0';
                    i++;
                }
                nums.push(x);
                i--;
            }   
            else{
                if(s[i] == '-' and(i==0 or(i>0 and s[i-1]=='('))){
                    nums.push(0);
                }
                parseOps(nums,ops,s[i]);
            } 
            i++;
        }
        while(!ops.empty()){
            calWithOp(nums,ops.top());
            ops.pop();
        }
        return nums.top();
    }
};
```

<hr>

## 71. Simplify Path
```
class Solution {
public:
    string simplifyPath(string path) {
        if(path.empty()) return path;
        path.push_back('/');
        vector<string> s;
        string curDirName;
        for(int i=0;i<path.size();i++){
            char c = path[i];
            if(c == '/'){
                if(i>0 and path[i-1] == '/') continue; 
                if(curDirName == "." or curDirName.empty()){}
                else if(curDirName == ".."){
                    if(!s.empty()) s.pop_back();
                } 
                else{
                    s.push_back(curDirName);
                } 
                curDirName.clear();
            }
            else{
                curDirName.push_back(c);
            }
        }
        string result;
        for(int i=0;i<s.size();i++){
            result+="/";
            result+=s[i];
        }
        return result.empty()?"/":result;
    }
};
```

<hr>

## 388. Longest Absolute File Path
`find_last_of()` will return -1 if not found
```
class Solution {
public:
   int lengthLongestPath(string input) {
        int result = 0;
        vector<string> lines;
        for(int left=0,i=0;i<input.size();i++){
            if(i == input.size()-1){
                lines.push_back(input.substr(left,i-left+1));
            }
            else if(input[i]=='\n'){
                lines.push_back(input.substr(left,i-left));
                left = i+1;
            }
        }
        stack<pair<int,int>> s;
        int curSize = 0;
        for(int i=0;i<lines.size();i++){
            int level = lines[i].find_last_of('\t')+1;
            cout<<level<<endl;
            while(!s.empty() and s.top().second >= level){
                curSize -= s.top().first;
                s.pop();
            }
            curSize += lines[i].size()-level;
            s.push({lines[i].size()-level,level});    
            if(lines[i].find(".") != string::npos)
                result = max(result,curSize+int(s.size())-1);
        }
        return result;
    }
};
```

<hr>

## 394. Decode String
#### Brute force Stack
```class Solution {
public:
    string decodeString(string s) {
        int index = 0;
        return helper(s,index,1);
    }
    string helper(const string&s,int&index,int repeat){
        string ret;
        int repeatCnt;
        string curStr;
        while(index<s.size()){
            char c = s[index];
            index++;
            if(c >= '0' and c <= '9'){
                repeatCnt = repeatCnt*10 + c - '0';
            }
            else if(c == '['){
                ret += helper(s,index,repeatCnt);
                repeatCnt = 0;
            }
            else if(c == ']'){
                break;
            }
            else{
                curStr.push_back(c);
                if(index == s.size() or !(s[index]>='a' and s[index]<='z' )){
                    ret+=curStr;
                    curStr.clear();
                }
            }
        }
        string tmp = ret;
        for(int i=0;i<repeat-1;i++){
            ret += tmp;   
        }
        return ret;
    }
};
```

<hr>

## 385. Mini Parser
```
class Solution {
public:
    NestedInteger deserialize(string s) {
        if(s.empty()) return NestedInteger();
        if(s[0] != '[') return NestedInteger(stoi(s));
        stack<NestedInteger> st;
        int start = 1;
        for(int i=0;i<s.size();i++){
            if(s[i] == '[') {
                st.push(NestedInteger());
                start = i+1;
            }else if(s[i] == ']' or s[i] == ','){
                //start might = ]
                if(i > start){
                    st.top().add(NestedInteger(stoi(s.substr(start, i - start))));
                }
                start=i+1;
                //marks the end of current level, add current NestedInteger to higher level
                //st.size > 1 means there are other nestedIntegers above
                if(s[i] == ']'){
                    if(st.size() > 1){
                        auto t = st.top(); st.pop();
                        st.top().add(t);
                    }
                }
            }
        }
        // At the end, there is only one element in the stack
        return st.top();
    }
};
```

<hr>


## Priority Queue Implementation
```
#include <iostream>
#include <vector>
using namespace std;

template <typename T>
class Heap{
private:
    int Size;
    int curSize;
    vector<T> arr;
public:
    Heap(int Size){
        this->Size = Size;
        curSize = 0;
        arr.reserve(Size+1);
    }
    void push(T x){
        if(curSize == Size){
            arr[Size] = x;
            swim(Size);
        }
        else{
            arr[curSize] = x;
            swim(curSize);
            curSize++;
        }
    }
    void swim(int pos){
        while(pos){
            int next = pos/2;
            if(arr[pos] < arr[next]) break;
            swap(arr[pos],arr[next]);
            pos = next;
        }
    }
    void sink(int pos){
        while(pos*2<Size){
            int j = 2*pos;
            if(j+1<Size and arr[j]<arr[j+1]) j++;
            if(arr[j] < arr[pos]) break;
            swap(arr[j],arr[pos]);
            pos = j;
        }
    }
    T top(){
        if(curSize!=0){
            return arr[0]; 
        }
        return -1;
    }
    T pop(){
        int ret = top();
        if(ret > 0){
            arr[0] = arr[curSize-1];
            curSize--;
            arr.pop_back();
            sink(0);
        }
        return ret;
    }
};
int main()
{
    Heap<int> h(5);
    
    h.push(1);
    h.push(2);
    h.push(3);
    h.push(4);
    h.push(5);
        
    cout<<h.top()<<endl;
     h.push(6);

    cout<<h.top()<<endl;
    h.pop();
    cout<<h.top()<<endl;
    h.pop();
    cout<<h.top()<<endl;

    return 0;
}

```

<hr>

## 215. Kth Largest Element in an Array
求最大值要用小根堆，不是大根堆
```
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        if(k == 0 or nums.size() == 0) return 0;
        // smaller items goes to the top
        priority_queue<int,vector<int>,greater<int>> q;
        for(int i=0;i<k;i++){
            q.push(nums[i]);
        }
        for(int i=k;i<nums.size();i++){
            if(nums[i] > q.top()){
                q.pop();
                q.push(nums[i]);
            }
        }
        return q.top();
    }
};
```

## 394. Decode String
```
class Solution:
    def decodeString(self, s: str) -> str:
        self.idx = 0
        def helper(s):
            cur = ""
            while self.idx < len(s):
                if s[self.idx] in "0123456789":
                    curNum = 0
                    while self.idx < len(s) and s[self.idx] in "0123456789":
                        curNum = curNum * 10 + int(s[self.idx])
                        self.idx += 1
                    #skip "["
                    self.idx+=1
                    cur += curNum * helper(s)
                elif s[self.idx] == "]":
                    self.idx+=1
                    break
                else:
                    cur += s[self.idx]
                    self.idx += 1
            return cur
        return helper(s)
        
```

## (lint)849. Basic Calculator III
## 224. Basic Calculator

https://www.lintcode.com/problem/basic-calculator-iii/description

```class Solution:
    def calculate(self, s: str) -> int:
        def level(c):
            if c in "+-": return 0
            elif c in "*/": return 1
            elif c in "(":return -2
            else:return -1
        def calWithOp(num_s,op):
            num2 = num_s.pop()
            num1 = num_s.pop()
            print(num1,num2,op)
            if op == "+": num_s.append(num1+num2)
            if op == "-": num_s.append(num1-num2)
            if op == "*": num_s.append(num1*num2)
            if op == "/": num_s.append(num1//num2)
            # print(num_s)
        s2 = ""
        for c in s:
            if c in "0123456789":
                s2+= c
            else: s2 += " "+c+" "
        tokens = s2.split()
        num_s = []
        op_s = []
        for token in tokens:
            if token[0] in "0123456789":
                num_s.append(int(token))
            else:
                op_level = level(token)
                # print(token,op_level)
                if op_level == -1:
                    while True:
                        cur_op = op_s.pop()
                        if level(cur_op) == -2:
                            break
                        else:
                            calWithOp(num_s,cur_op)
                elif op_level == -2:
                    op_s.append(token)
                elif len(op_s) == 0 or level(token) > level(op_s[-1]):
                    op_s.append(token)
                else:
                    while op_s and level(token) <= level(op_s[-1]):
                        cur_op = op_s.pop()
                        calWithOp(num_s,cur_op)
                    op_s.append(token)
            # print(token,num_s)
        while op_s:
            calWithOp(num_s,op_s.pop())
        return num_s[-1]
```